"""Hard-only in-process GPU entry point. Verify held FDs before any CUDA-capable import."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import runpy
import sys

from tools import run_colab_hard as hard


class SmokeAudit:
    """Observe actual scaler batches and optimizer calls without changing batch/schedule/EMA behavior."""

    def __init__(self):
        self.losses = []
        self.update_batches = []
        self.devices = set()

    def __enter__(self):
        import torch
        from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
        self.original_scale = torch.amp.GradScaler.scale

        def scale(scaler, loss, *args, **kwargs):
            hard.require(torch.is_tensor(loss) and loss.numel() == 1 and hard.finite_tree(loss),
                         'Smoke nonfinite or nonscalar loss')
            self.losses.append(float(loss.detach()))
            self.devices.add(loss.device.type)
            return self.original_scale(scaler, loss, *args, **kwargs)

        def before(optimizer, args, kwargs):
            grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            hard.require(isinstance(optimizer, torch.optim.AdamW) and grads and hard.finite_tree(grads),
                         'Smoke successful step requires finite unscaled gradients')

        def after(optimizer, args, kwargs):
            hard.require(hard.finite_tree(optimizer.state_dict()) and
                         hard.finite_tree([p for g in optimizer.param_groups for p in g['params']]),
                         'Smoke nonfinite optimizer update')
            self.update_batches.append(len(self.losses))

        self.pre_hook = register_optimizer_step_pre_hook(before)
        self.post_hook = register_optimizer_step_post_hook(after)
        torch.amp.GradScaler.scale = scale
        return self

    def __exit__(self, *exc):
        import torch
        torch.amp.GradScaler.scale = self.original_scale
        self.pre_hook.remove()
        self.post_hook.remove()

    def report(self):
        return dict(processed_batches=len(self.losses), successful_optimizer_updates=len(self.update_batches),
                    skipped_optimizer_updates=len(self.losses) - len(self.update_batches),
                    successful_update_batches=self.update_batches, losses=self.losses,
                    devices=sorted(self.devices), finite_successful_gradients_and_state=True)


def training_command(cfg, repo):
    from il.train import _demo_paths, _flags_to_cli
    from omegaconf import OmegaConf
    hard.require(cfg.baseline_dir == 'diffusion_policy' and cfg.script == 'train_rgbd.py'
                 and cfg.demo_dir == 'hard' and cfg.max_episode_steps == 800, 'Hard trainer contract differs')
    demos = _demo_paths(cfg)
    hard.require(demos == [str(Path(repo) / hard.DEMO)], 'Hard demo path differs')
    flags = OmegaConf.to_container(cfg.flags, resolve=True)
    common = ['--env-id', cfg.env_id, '--control-mode', cfg.control_mode,
              '--sim-backend', cfg.sim_backend, '--max-episode-steps', str(cfg.max_episode_steps)]
    return [sys.executable, '-u', cfg.script, '--demo-path', *demos] + common + _flags_to_cli(flags)


def run_training(cfg, repo, local, fds):
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    hard.require(HydraConfig.get().runtime.choices['method'] == 'dp_rgb_hard', 'Hard method required')
    cmd = training_command(cfg, repo)
    hard.require(all(Path(p).is_file() for p in [str(Path(repo) / hard.DEMO)]), 'Staged demo missing')
    output = Path(HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'resolved-config.yaml', resolve=True)
    hard.write_json(output / 'launcher.json', dict(command=cmd, hydra_overrides=list(HydraConfig.get().overrides.task),
                    pid=os.getpid(), lock_paths=[str(p) for p in hard.lock_paths(repo, local)]))
    cwd = Path(repo) / 'il/baselines/diffusion_policy'
    print(f'[il/train] method=dp_rgb_hard\n[il/train] cwd={cwd}\n[il/train] {" ".join(cmd)}', flush=True)
    print('[hard] executing trainer in this lock-owning process', flush=True)
    smoke = cfg.flags.total_iters == 100
    if smoke:
        hard.require(cfg.flags.eval_freq == 0 and cfg.flags.log_freq == 1, 'Smoke logging/eval contract differs')
    audit = SmokeAudit() if smoke else nullcontext()
    # Repeat at the boundary: no subprocess/exec/close_fds hop between this check and CUDA.
    hard.verify_inherited_locks(fds, repo, local)
    with audit:
        execute_script(cwd / cfg.script, cmd[3:], cwd)
    if smoke:
        hard.write_json(Path(local) / 'smoke/batch-audit.json', audit.report())


def execute_script(path, args, cwd):
    old_cwd, old_argv, old_path = Path.cwd(), sys.argv, sys.path[:]
    try:
        os.chdir(cwd)
        sys.argv = [str(path), *args]
        sys.path.insert(0, str(Path(path).parent))
        runpy.run_path(str(path), run_name='__main__')
    finally:
        os.chdir(old_cwd)
        sys.argv, sys.path[:] = old_argv, old_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--lock-fds', required=True, nargs=3, type=int)
    parser.add_argument('entry', choices=['il/train.py', 'eval.py', 'tools/rollout_logger.py', 'tools/run_colab_hard.py'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    hard.verify_inherited_locks(args.lock_fds, args.repo, args.local)
    if args.entry == 'il/train.py':
        import hydra
        sys.argv = [sys.argv[0], *args.arguments]
        @hydra.main(version_base=None, config_path=str(Path(args.repo) / 'il/conf'), config_name='train')
        def launch(cfg):
            run_training(cfg, args.repo, args.local, args.lock_fds)
        launch()
    else:
        execute_script(Path(args.repo) / args.entry, args.arguments, args.repo)


if __name__ == '__main__':
    main()
