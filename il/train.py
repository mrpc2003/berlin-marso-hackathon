"""Hydra dispatcher for the IL baselines (consistent with the repo's Hydra style).

This does NOT reimplement any learning logic. It loads a method config (il/conf/method/*.yaml),
converts its `flags` into the vendored baseline script's CLI (underscore keys -> --hyphen-flags;
booleans -> --flag / --no-flag), and runs that real script with the right demo path + shared args.

  python il/train.py method=dp_rgb_easy                     # rgb DP, easy demos, 250-step eval
  python il/train.py method=dp_rgb demo_dir=[hard,medium]   # mixed-level training (first = eval scene)
  pixi run python il/train.py method=dp_rgb flags.total_iters=8000 flags.eval_freq=4000

Outputs (checkpoints + tensorboard + eval videos) land under
  il/baselines/<baseline_dir>/runs/<flags.exp_name>/
Evaluate via: pixi run python eval.py ... policy=warehouse_sort.il_policy:load_dp_rgb ...
"""

import os
import subprocess
import sys

import hydra
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))


def _demo_path(demo_dir, kind):
    d = os.path.join(HERE, "demos", demo_dir)
    return os.path.join(d, f"trajectory.{kind}.pd_ee_delta_pos.physx_cuda.h5")


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _demo_paths(cfg):
    """demo_path (str | list) wins; else one file per demo_dir entry (str | list, e.g. demo_dir=[hard,medium])."""
    paths = _as_list(cfg.get("demo_path"))
    if not paths:
        paths = [_demo_path(d, cfg.demo_kind) for d in _as_list(cfg.get("demo_dir", "easy"))]
    return paths


def _flags_to_cli(flags: dict):
    """method.flags (underscore keys) -> vendored tyro CLI args.
    bool True -> --flag, bool False -> --no-flag; everything else -> --flag value."""
    cli = []
    for key, val in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            cli.append(flag if val else "--no-" + key.replace("_", "-"))
        elif isinstance(val, (list, tuple)):
            cli += [flag] + [str(v) for v in val]
        elif val is None:
            continue
        else:
            cli += [flag, str(val)]
    return cli


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg):
    demos = _demo_paths(cfg)
    for demo in demos:
        if not os.path.exists(demo):
            sys.exit(f"demo dataset not found: {demo}\n  run: python il/download_demos.py  (or il/gen_demos.py)")
    common = ["--env-id", cfg.env_id, "--control-mode", cfg.control_mode,
              "--sim-backend", cfg.sim_backend, "--max-episode-steps", str(cfg.max_episode_steps)]
    flags = OmegaConf.to_container(cfg.flags, resolve=True)
    cmd = [sys.executable, "-u", cfg.script, "--demo-path", *demos] + common + _flags_to_cli(flags)
    cwd = os.path.join(HERE, "baselines", cfg.baseline_dir)
    method = hydra.core.hydra_config.HydraConfig.get().runtime.choices.get("method", cfg.script)
    print(f"[il/train] method={method}\n"
          f"[il/train] cwd={cwd}\n[il/train] {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.run(cmd, cwd=cwd).returncode)


if __name__ == "__main__":
    main()
