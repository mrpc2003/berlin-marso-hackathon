"""Bounded, local-first Hard RGB DP supervisor. Reuses the staged Colab runtime.

No installs, dataset transfers, resume inputs, submissions, or service management.
The notebook embeds this stdlib-only module for guarded source bootstrap.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid

RUNTIME = dict(repo='/content/berlin-marso-hackathon', python='/content/marso-py312/bin/python',
               drive='/content/drive/MyDrive/marso', mount='/content/drive', local='/content/marso-hard')
DEMO = 'il/demos/hard/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5'
PRESET = dict(total_iters=40000, batch_size=128, lr=1e-4, obs_mode='rgb', obs_camera='scene',
              visual_encoder='resnet18', obs_horizon=2, act_horizon=8, pred_horizon=16,
              amp=True, num_demos=None, seed=1, save_freq=1000, eval_freq=5000,
              num_eval_episodes=32, num_eval_envs=8, skip_initial_eval=True,
              capture_video=False, exp_name_timestamp=False, resume=None)
RANDOMIZATION = {'parcel_pose': {'xy_jitter': [-0.02, 0.02], 'yaw_jitter': [-0.1, 0.1]},
                 'bin_position': {'side_swap_prob': 0.5, 'xy_jitter': [0.0, 0.0]}}
SWEEP = ((8, 16), (4, 16), (8, 32), (4, 32))
TRAIN_TIMEOUT = 6 * 3600
PROGRESS_SECONDS = 60
SYNC_TIMEOUT = 45
FINAL_SYNC_TIMEOUT = 600
# Explicit reviewed payload contract: additions require a source review, never a Git lookup at runtime.
SOURCE_FILES = ('eval.py', 'il/train.py', 'examples/scripted_policy.py', 'tools/rollout_logger.py',
                'tools/run_colab_hard.py', 'tools/make_colab_hard_notebook.py',
                'tools/run_colab_medium_resume.py', 'tools/hard_gpu_worker.py',
                'conf/config.yaml', 'conf/difficulty/easy.yaml', 'conf/difficulty/hard.yaml',
                'conf/difficulty/medium.yaml', 'conf/eval/default.yaml', 'conf/eval/eval32.yaml',
                'conf/eval/eval64.yaml', 'il/conf/train.yaml', 'il/conf/method/act_rgb.yaml',
                'il/conf/method/dp.yaml', 'il/conf/method/dp_rgb.yaml', 'il/conf/method/dp_rgb_easy.yaml',
                'il/conf/method/dp_rgb_hard.yaml', 'il/conf/method/dp_rgb_medium.yaml',
                'warehouse_sort/__init__.py', 'warehouse_sort/act_policy.py', 'warehouse_sort/constants.py',
                'warehouse_sort/env.py', 'warehouse_sort/il_policy.py', 'warehouse_sort/utils.py',
                'il/baselines/diffusion_policy/train_rgbd.py', 'il/baselines/diffusion_policy/train.py',
                'il/baselines/diffusion_policy/setup.py', 'il/baselines/diffusion_policy/record_dp.py',
                'il/baselines/diffusion_policy/diffusion_policy/__init__.py',
                'il/baselines/diffusion_policy/diffusion_policy/augment.py',
                'il/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py',
                'il/baselines/diffusion_policy/diffusion_policy/evaluate.py',
                'il/baselines/diffusion_policy/diffusion_policy/lerobot_encoder.py',
                'il/baselines/diffusion_policy/diffusion_policy/make_env.py',
                'il/baselines/diffusion_policy/diffusion_policy/plain_conv.py',
                'il/baselines/diffusion_policy/diffusion_policy/streaming_dataset.py',
                'il/baselines/diffusion_policy/diffusion_policy/utils.py')
MACHINE_GPU_LOCK = Path('/tmp/marso-active-gpu.lock')


def validate_source_set(payload):
    require(isinstance(payload, dict) and set(payload) == set(SOURCE_FILES),
            'Reviewed source set mismatch (omissions or unexpected payloads)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def no_links(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, f'Unsafe path: {path}')
    for part in (path, *path.parents):
        require(not part.is_symlink(), f'Symlink rejected: {part}')
    return path


def child(root, rel):
    rel = Path(rel)
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel) != '.', f'Unsafe relative path: {rel}')
    return no_links(Path(root) / rel)


def write_json(path, obj):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with tmp.open('x') as stream:
            json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def roots(exp, scope=RUNTIME):
    require(isinstance(exp, str) and re.fullmatch(r'hard_[0-9]{8}_[0-9]{6}', exp) is not None,
            'Expected new hard_YYYYMMDD_HHMMSS experiment')
    local = no_links(Path(scope['local']) / exp)
    remote = no_links(Path(scope['drive']) / 'recoveries/hard' / exp)
    return local, remote


def ensure_drive(scope=RUNTIME):
    mount = no_links(Path(scope['mount']))
    root = no_links(Path(scope['drive']))
    require(mount.is_mount(), f'Real Drive mount required: {mount}')
    require(root.is_dir() and root.is_relative_to(mount), f'Existing Drive root required: {root}')
    # No Hard demo marker: data is staged independently in the repository by the parent.


def verify_sources(repo, hashes):
    validate_source_set(hashes)
    for rel, digest in hashes.items():
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid source digest')
        require(sha(child(repo, rel)) == digest, f'Source changed: {rel}')


def reject_gpu_processes():
    rows = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True, timeout=10)
    for row in rows.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        if re.search(r'(?:^|\s|/)(?:train_rgbd|train|eval|rollout_logger|scripted_policy)\.py(?:\s|$)', command):
            raise RuntimeError(f'Train/eval process already active (pid {fields[0]})')
        if re.search(r'(?:^|\s|/)run_colab_\w+\.py(?:\s|$)', command) and '--sync-worker' not in command:
            raise RuntimeError(f'Colab supervisor already active (pid {fields[0]})')


@contextmanager
def launch_locks(repo, local):
    """Machine-wide contract shared with package/generalization, then optional narrower locks."""
    handles = []
    try:
        for path in lock_paths(repo, local):
            no_links(path)
            handle = path.open('a')
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Launch lock busy: {path}') from exc
        reject_gpu_processes()
        yield tuple(handle.fileno() for handle in handles)
    finally:
        for handle in reversed(handles):
            handle.close()


def lock_paths(repo, local):
    # macOS aliases /tmp to /private/tmp. Resolve only the trusted parent, never the lock itself.
    return (MACHINE_GPU_LOCK.parent.resolve() / MACHINE_GPU_LOCK.name,
            Path(repo) / '.marso-gpu-launch.lock', Path(local) / '.experiment.lock')


def verify_inherited_locks(fds, repo, local):
    require(len(fds) == 3 and len(set(fds)) == 3, 'Worker requires three inherited supervisor launch locks')
    for fd, path in zip(fds, lock_paths(repo, local)):
        actual, expected = os.fstat(fd), no_links(path).stat()
        require((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'Invalid inherited lock')
        # Inode equality alone accepts an unlocked FD. An independent open must be blocked,
        # while re-locking the inherited open-file description must succeed without acquiring anew.
        with path.open('a') as contender:
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError('Inherited lock is not already held')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Inherited FD does not own the lock') from exc


def bootstrap(bundle, exp, scope=RUNTIME):
    """Validate every base/current hash BEFORE changing any reviewed runtime source."""
    repo = no_links(Path(scope['repo']))
    require(repo.is_dir() and Path(scope['python']).is_file(), 'Reuse existing repo and Python 3.12 runtime')
    local, remote = roots(exp, scope)
    ensure_drive(scope)
    require(not local.exists() and not remote.exists(), 'Refusing to reuse an existing experiment directory')
    validate_source_set(bundle)
    for rel, entry in bundle.items():
        target = child(repo, rel)
        payload = base64.b64decode(entry['content_b64'], validate=True)
        require(hashlib.sha256(payload).hexdigest() == entry['sha256'], f'Corrupt embedded source: {rel}')
        actual = sha(target) if target.exists() else None
        require(actual in (entry['base_sha256'], entry['sha256']), f'Unexpected runtime source: {rel}')
    # Both roots must be NEW, including empty roots. No existing output may be adopted.
    local.mkdir(parents=True, exist_ok=False)
    with launch_locks(repo, local):
        ensure_drive(scope)
        remote.mkdir(parents=True, exist_ok=False)
        for rel, entry in bundle.items():
            target = child(repo, rel)
            payload = base64.b64decode(entry['content_b64'], validate=True)
            snapshot = child(local / 'sources', rel)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(payload)
            actual = sha(target) if target.exists() else None
            require(actual in (entry['base_sha256'], entry['sha256']), f'Source changed during setup: {rel}')
            if actual != entry['sha256']:
                if target.exists():
                    backup = child(local / 'source-backups', rel)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                    require(sha(backup) == actual, 'Backup verification failed')
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.tmp.' + uuid.uuid4().hex)
                with tmp.open('xb') as stream:
                    stream.write(payload)
                os.replace(tmp, target)
        hashes = {rel: entry['sha256'] for rel, entry in bundle.items()}
        verify_sources(repo, hashes)
        plan = dict(exp=exp, repo=str(repo), python=scope['python'], drive_root=scope['drive'],
                    local_root=str(local), level='hard', method='dp_rgb_hard', max_episode_steps=800,
                    num_parcels=6, preset=PRESET, source_sha256=hashes)
        write_json(local / 'run/source-install.json', {rel: {k: v for k, v in e.items() if k != 'content_b64'}
                                                     for rel, e in bundle.items()})
        write_json(local / 'run/plan.json', plan)
        write_json(local / 'run/status.json', dict(status='prepared', compute_status='not_started',
                                                   remote_status='pending', remote_verified=False))
    return local / 'run/plan.json'


def load_plan(path, scope=RUNTIME):
    path = no_links(Path(path))
    plan = json.loads(path.read_text())
    local, remote = roots(plan.get('exp'), scope)
    require(path == local / 'run/plan.json', 'Plan outside experiment')
    for key, expected in dict(repo=scope['repo'], python=scope['python'], drive_root=scope['drive'],
                              local_root=str(local), level='hard', method='dp_rgb_hard',
                              max_episode_steps=800, num_parcels=6, preset=PRESET).items():
        require(plan.get(key) == expected, f'Unexpected plan {key}')
    verify_sources(Path(plan['repo']), plan.get('source_sha256'))
    for rel, digest in plan['source_sha256'].items():
        require(sha(child(local / 'sources', rel)) == digest, f'Source snapshot changed: {rel}')
    return plan


def metadata_contract(meta):
    info = meta['env_info']
    kwargs = info['env_kwargs']
    require(info['env_id'] == 'WarehouseSort-v1', 'Unexpected dataset env')
    require(kwargs['num_parcels'] == 6 and kwargs['fixed_poses'] is False, 'Dataset must contain Hard six-parcel scenes')
    require(kwargs.get('obs_mode') == 'rgb' and kwargs.get('obs_camera', 'scene') == 'scene', 'Dataset must be scene RGB')
    require(kwargs.get('control_mode') == 'pd_ee_delta_pos', 'Unexpected control mode')
    if 'max_episode_steps' in kwargs:
        require(kwargs['max_episode_steps'] == 800, 'Dataset episode budget must be 800')
    for group, fields in RANDOMIZATION.items():
        for name, expected in fields.items():
            require(kwargs['randomization'][group][name] == expected, f'Wrong dataset randomization: {group}.{name}')
    return kwargs


def dataset_probe(path):
    import h5py
    import numpy as np
    path = no_links(Path(path))
    meta_path = no_links(path.with_suffix('.json'))
    before = {p.name: sha(p) for p in (path, meta_path)}
    meta = json.loads(meta_path.read_text())
    kwargs = metadata_contract(meta)
    lengths = []
    with h5py.File(path, 'r') as f:
        keys = sorted(f.keys())
        require(len(keys) == 200 and all(re.fullmatch(r'traj_\d+', k) for k in keys), 'Expected all 200 trajectories')
        for key in keys:
            g = f[key]
            a = g['actions'][()]
            n = a.shape[0]
            require(0 < n <= 800 and a.shape == (n, 4) and np.isfinite(a).all(), f'Invalid actions: {key}')
            lengths.append(n)
            for name, dim in (('obs/agent/qpos', 9), ('obs/agent/qvel', 9), ('obs/extra/tcp_pose', 7)):
                x = g[name][()]
                require(x.shape == (n + 1, dim) and np.isfinite(x).all(), f'Invalid finite state: {key}/{name}')
            grasp = g['obs/extra/is_grasped'][()]
            require(grasp.shape in ((n + 1,), (n + 1, 1)) and np.isfinite(grasp).all(), f'Invalid grasp state: {key}')
            rgb = g['obs/sensor_data/scene_camera/rgb']
            require(rgb.shape == (n + 1, 128, 128, 3) and rgb.dtype == np.uint8, f'Invalid RGB schema: {key}')
            # Read/decompress every frame with bounded memory (not just HDF5 metadata).
            for start in range(0, n + 1, 32):
                require(rgb[start:start + 32].shape[1:] == (128, 128, 3), f'Unreadable RGB: {key}')
    require(before == {p.name: sha(p) for p in (path, meta_path)}, 'Dataset changed during validation')
    return dict(hashes=before, h5_trajectories=200, json_episode_records=len(meta.get('episodes', [])),
                metadata_episodecount_mismatch=len(meta.get('episodes', [])) != 200,
                env_kwargs=kwargs, recorded_env_info=meta['env_info'], lengths=lengths, state_dim=26, action_dim=4,
                rgb_shape=[128, 128, 3], rgb_dtype='uint8', finite_state=True)


def gpu_probe():
    import torch
    import torchvision, mani_skill, gymnasium, diffusers, h5py, hydra, tyro, mplib  # noqa: F401
    from importlib.metadata import version
    require(sys.version_info[:2] == (3, 12), 'Reuse validated Python 3.12')
    versions = {n: version(n) for n in ('torch', 'torchvision', 'mani-skill', 'gymnasium', 'diffusers', 'mplib', 'numpy')}
    for name, expected in {'torch': '2.11.0', 'torchvision': '0.26.0', 'mani-skill': '3.0.1',
                           'gymnasium': '1.3.0', 'diffusers': '0.38.0', 'mplib': '0.1.1'}.items():
        require(versions[name].split('+')[0] == expected, f'Reuse environment mismatch: {name}')
    require(int(versions['numpy'].split('.')[0]) < 2, 'Validated runtime requires numpy < 2')
    require(torch.cuda.is_available(), 'CUDA is required; no CPU fallback')
    x = torch.randn(128, 128, device='cuda', requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    require(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item(), 'Nonfinite CUDA forward/backward')
    return dict(device='cuda', gpu=torch.cuda.get_device_name(), versions=versions,
                loss=float(loss), gradient_norm=float(x.grad.norm()), python=sys.version)


def reference_time_limit(env, cfg):
    """Verify the active wrapper budget; spec and base fields are metadata only."""
    try:
        wrapper_limit = env.get_wrapper_attr('_max_episode_steps')
    except AttributeError as exc:
        raise ValueError('Actual wrapper step limit missing') from exc
    require(type(wrapper_limit) is int and wrapper_limit == 800, 'Actual wrapper step limit != 800')
    require(cfg.max_episode_steps == 800 and cfg.difficulty.max_episode_steps == 800,
            'Configured Hard step limit != 800')
    return dict(max_episode_steps=wrapper_limit, wrapper_max_episode_steps=wrapper_limit,
                configured_max_episode_steps=cfg.max_episode_steps,
                difficulty_max_episode_steps=cfg.difficulty.max_episode_steps,
                base_max_episode_steps=getattr(env.unwrapped, 'max_episode_steps', None),
                spec_max_episode_steps=getattr(env.spec, 'max_episode_steps', None))


def reference_probe(repo):
    import numpy as np
    import torch
    import gymnasium as gym
    from omegaconf import OmegaConf
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from warehouse_sort.utils import compose_cfg
    from examples.scripted_policy import scripted_episode
    cfg = compose_cfg(['difficulty=hard', 'obs_mode=rgb'], str(Path(repo) / 'conf'))
    env = gym.make('WarehouseSort-v1', num_envs=1, obs_mode='rgb', obs_camera='scene',
                   control_mode='pd_ee_delta_pos', sim_backend='gpu', render_mode='rgb_array',
                   max_episode_steps=800, num_parcels=cfg.difficulty.num_parcels,
                   fixed_poses=cfg.difficulty.fixed_poses,
                   randomization=OmegaConf.to_container(cfg.randomization, resolve=True))
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    try:
        base = env.unwrapped
        require(base.num_parcels == 6 and len(base.parcels) == 6 and not base.fixed_poses, 'Actual env must be Hard')
        time_limit = reference_time_limit(env, cfg)
        require(base.device.type == 'cuda', 'Reference must use GPU physics')
        require(base._rand['parcel_xy_jitter'] == [-0.02, 0.02] and
                base._rand['parcel_yaw_jitter'] == [-0.1, 0.1] and base._rand['bin_side_swap_prob'] == 0.5,
                'Actual randomization differs from Hard')
        observations = []
        poses, sides = [], []
        # Fixed bounded resets provide observed pose jitter and both bin arrangements.
        for seed in range(5000, 5016):
            obs, _ = env.reset(seed=seed)
            rgb = obs['rgb']
            require(tuple(rgb.shape) == (1, 128, 128, 3) and rgb.dtype == torch.uint8, 'Actual scene RGB contract')
            poses.append(torch.stack([p.pose.raw_pose[0] for p in base.parcels]).cpu().tolist())
            sides.append(float(base.bins[0].pose.p[0, 1].item()))
            obs, reward, terminated, truncated, info = env.step(torch.zeros((1, 4), device=base.device))
            require(tuple(obs['state'].shape) == (1, 26) and torch.isfinite(obs['state']).all().item(),
                    'Nonfinite or incorrect reference proprioception')
            require(tuple(obs['rgb'].shape) == (1, 128, 128, 3), 'Step RGB contract')
            observations.append(dict(seed=seed, reward=float(reward.item()), sorted=int(info['success_count'].item())))
        require(not np.allclose(poses[0], poses[1]), 'No observed parcel pose randomization')
        require(min(sides) < 0 < max(sides), 'Both bin arrangements not observed in 16 bounded resets')
        frame = env.render()
        frame = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
        require(frame.ndim in (3, 4) and frame.shape[-1] in (3, 4) and np.isfinite(frame).all(), 'Invalid render')
        hist = scripted_episode(env, max_steps=800, seed=5000)
        require(len(hist) > 0, 'Reference script produced no steps')
        count = int(hist[-1][-1]['success_count'].item())
        require(0 <= count <= 6, 'Invalid reference count')
        return dict(level='hard', num_parcels=base.num_parcels, **time_limit,
                    device=str(base.device), rgb_shape=[1, 128, 128, 3], render_shape=list(frame.shape),
                    resets=observations, parcel_poses=poses, red_bin_y=sides, actual_randomization=base._rand,
                    scripted_seed=5000, scripted_steps=len(hist), scripted_sorted=count,
                    scripted_success=count == 6, reference_valid=True)
    finally:
        env.close()


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_checkpoint(ck, total=40000, *, final=True):
    import torch
    required = {'agent', 'ema_agent', 'optimizer', 'lr_scheduler', 'ema_state', 'scaler', 'rng',
                'args', 'config', 'iteration', 'best_eval_metrics', 'eval_history'}
    require(required <= ck.keys(), 'Missing durable training state')
    iteration = ck['iteration']
    require(iteration == total if final else 0 < iteration <= total, 'Wrong checkpoint iteration')
    args, cfg = ck['args'], ck['config']
    for key in ('batch_size', 'lr', 'obs_horizon', 'act_horizon', 'pred_horizon', 'obs_mode', 'visual_encoder'):
        require(args.get(key) == PRESET[key] and cfg.get(key) == PRESET[key], f'Checkpoint contract: {key}')
    for key in ('obs_camera', 'amp', 'seed', 'num_demos', 'resume'):
        require(key in args and args[key] == PRESET[key], f'Checkpoint args: {key}')
    require(args['total_iters'] == cfg['total_iters'] == total, 'Wrong training target')
    require(args['max_episode_steps'] == cfg['max_episode_steps'] == 800, 'Wrong checkpoint step limit')
    require(cfg['state_dim'] == 26 and cfg['act_dim'] == 4 and cfg['image_hw'] == [128, 128], 'Wrong policy observation contract')
    require(cfg['demo_paths'] == [str(Path(RUNTIME['repo']) / DEMO)], 'Checkpoint demo provenance differs')
    # Metadata overrides legacy args.num_parcels in train_rgbd; dataset/reference gates prove six.
    for key in ('agent', 'ema_agent'):
        require(bool(ck[key]) and all(torch.is_tensor(t) for t in ck[key].values()) and finite_tree(ck[key]), f'Invalid {key}')
    opt = ck['optimizer']
    require(bool(opt.get('state')) and bool(opt.get('param_groups')) and finite_tree(opt), 'Invalid optimizer state')
    # enumerate starts at 0 and takes total_iters batches. Final is relabelled total_iters.
    steps = iteration if iteration == total else iteration + 1
    require(ck['lr_scheduler'].get('last_epoch') == steps and finite_tree(ck['lr_scheduler']), 'Scheduler step mismatch')
    require(ck['ema_state'].get('optimization_step') == steps and bool(ck['ema_state'].get('shadow_params'))
            and all(torch.is_tensor(t) for t in ck['ema_state']['shadow_params'])
            and finite_tree(ck['ema_state']), 'Invalid EMA training state')
    require(bool(ck['scaler']) and finite_tree(ck['scaler']), 'AMP scaler state missing')
    for key in ('save_freq', 'num_eval_episodes', 'num_eval_envs', 'skip_initial_eval', 'capture_video', 'exp_name_timestamp'):
        require(args.get(key) == PRESET[key], f'Checkpoint training settings: {key}')
    require(args.get('eval_freq') == (0 if total == 100 else 5000), 'Wrong training eval frequency')
    if total == 40000:
        require([r['iteration'] for r in ck['eval_history']] == list(range(5000, iteration + 1, 5000))
                and all(r['n_episodes'] == 32 and finite_tree(r) and 0 <= r['sort_accuracy'] <= 1
                        for r in ck['eval_history']),
                'Missing actual training-time evaluation evidence')
    rng = ck['rng']
    require(isinstance(rng.get('python'), tuple) and bool(rng['python'])
            and isinstance(rng.get('numpy'), tuple) and bool(rng['numpy']), 'Missing Python/numpy RNG')
    require(isinstance(rng.get('cuda'), (list, tuple)) and bool(rng['cuda']), 'Missing CUDA RNG')
    for state in [rng.get('torch'), *rng['cuda']]:
        require(torch.is_tensor(state) and state.device.type == 'cpu' and state.dtype == torch.uint8
                and state.ndim == 1 and state.numel() > 0, 'Invalid RNG tensor')
    updates = [float(s.get('step', -1)) for s in opt['state'].values()]
    require(all(math.isfinite(x) and x.is_integer() and 0 < x <= steps for x in updates),
            'Invalid successful optimizer update count')
    if total == 100:
        require(len(set(updates)) == 1, 'Smoke optimizer update counts disagree')
    return dict(iteration=iteration, scheduler_step=steps, finite_agent_and_ema=True,
                processed_batches=steps, successful_optimizer_updates=int(min(updates)),
                skipped_optimizer_updates=steps - int(min(updates)),
                optimizer_state_entries=len(opt['state']), total_iters=total,
                best_sort_accuracy=ck['best_eval_metrics'].get('sort_accuracy', 0.0),
                evaluation_history=ck['eval_history'])


def validate_smoke_audit(audit, report, device='cuda'):
    require(finite_tree(audit), 'Nonfinite smoke audit')
    require(audit.get('processed_batches') == report['processed_batches'] == 100, 'Smoke requires 100 processed batches')
    losses, batches = audit.get('losses', []), audit.get('successful_update_batches', [])
    updates = audit.get('successful_optimizer_updates')
    require(len(losses) == 100 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses),
            'Smoke requires 100 finite losses')
    require(isinstance(updates, int) and 0 < updates <= 100 and updates == report['successful_optimizer_updates']
            and len(batches) == updates and batches == sorted(set(batches))
            and all(type(i) is int and 1 <= i <= 100 for i in batches), 'Smoke successful update evidence differs')
    require(audit.get('skipped_optimizer_updates') == report['skipped_optimizer_updates'] == 100 - updates,
            'Smoke skipped update evidence differs')
    require(audit.get('devices') == [device] and audit.get('finite_successful_gradients_and_state') is True,
            'Smoke device/finite gradient evidence missing')
    return dict(loss_count=100, loss_min=min(losses), loss_max=max(losses), **{
        k: audit[k] for k in ('processed_batches', 'successful_optimizer_updates', 'skipped_optimizer_updates')})


def checkpoint_report(path, total=40000, *, final=True):
    import torch
    path = no_links(Path(path))
    before = sha(path)
    result = validate_checkpoint(torch.load(path, map_location='cpu', weights_only=False), total, final=final)
    require(sha(path) == before, 'Checkpoint changed while validating')
    return dict(path=str(path), sha256=before, **result)


def train_command(plan, smoke=False):
    local = Path(plan['local_root'])
    label = 'smoke' if smoke else 'train'
    return [plan['python'], '-u', 'il/train.py', 'method=dp_rgb_hard',
            f'flags.exp_name={local / label}', f'flags.ckpt_dir={local / label / "ckpts"}',
            'flags.resume=null', 'flags.num_demos=null', 'flags.total_iters=' + ('100' if smoke else '40000'),
            'flags.save_freq=1000', 'flags.eval_freq=' + ('0' if smoke else '5000'),
            'flags.num_eval_episodes=32', 'flags.num_eval_envs=8', 'flags.skip_initial_eval=true',
            '+flags.seed=1', 'flags.capture_video=false', 'flags.exp_name_timestamp=false',
            'flags.log_freq=' + ('1' if smoke else '500'), f'hydra.run.dir={local / label / "hydra"}']


def eval_command(plan, checkpoint, label, n, horizon, steps, *, video=False, diagnostic=False):
    local = Path(plan['local_root'])
    folder = child(local, f'evaluations/{label}')
    folder.mkdir(parents=True, exist_ok=False)
    config = folder / 'seeds.yaml'
    config.write_text('eval:\n  n_episodes: %d\n  seeds: %s\n' % (n, json.dumps(list(range(5000, 5000 + n)))))
    cmd = [plan['python'], '-u', 'tools/rollout_logger.py' if diagnostic else 'eval.py',
           'difficulty=hard', 'obs_mode=rgb', 'policy=warehouse_sort.il_policy:load_dp_rgb',
           f'checkpoint={checkpoint}', f'eval_config={config}', f'num_envs={1 if video else 8}',
           'record_video=' + str(video).lower(), 'video_envs=1', f'hydra.run.dir={folder}',
           f'+policy_kwargs.act_horizon={horizon}', f'+policy_kwargs.num_inference_steps={steps}']
    if diagnostic:
        cmd += ['+log_episodes=8', f'+log_out={folder / "diagnostics.json"}']
    else:
        cmd += [f'results_file={folder / "metrics.jsonl"}']
    return cmd, folder


def validate_eval(row, checkpoint, digest, n, horizon, steps):
    for key, expected in dict(level='hard', obs_mode='rgb', n_episodes=n, requested_n_episodes=n,
                              num_parcels=6, max_episode_steps=800, seed0=5000,
                              checkpoint=str(checkpoint), policy_kwargs=dict(act_horizon=horizon, num_inference_steps=steps)).items():
        require(row.get(key) == expected, f'Wrong actual eval {key}')
    require(finite_tree(row), 'Nonfinite evaluation output')
    for key, high in (('sort_accuracy', 1), ('all_placed_rate', 1), ('mis_sort_rate', 1),
                      ('mean_sorted', 6), ('mean_steps', 800), ('eval_seconds', math.inf)):
        numeric_range(row.get(key), 0, high, f'evaluation {key}')
    require(math.isclose(row['mean_sorted'], row['sort_accuracy'] * 6, abs_tol=1e-6), 'Inconsistent evaluation sort metrics')
    require(sha(no_links(Path(checkpoint))) == digest, 'Evaluation checkpoint changed')
    return dict(row, checkpoint_sha256=digest, score_scope='local validation seeds; not official or heldout')


def numeric_range(value, low, high, label, integer=False):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high
            and (not integer or int(value) == value), f'Invalid {label} range/type')


def validate_diagnostics(folder, checkpoint, digest):
    import numpy as np
    folder = Path(folder)
    json_path, npz_path = folder / 'diagnostics.json', folder / 'diagnostics_traces.npz'
    before = {p.name: sha(no_links(p)) for p in (json_path, npz_path)}
    diag = json.loads(json_path.read_text())
    require(finite_tree(diag), 'Nonfinite diagnostic JSON')
    require(diag['level'] == 'hard' and diag['checkpoint'] == str(checkpoint) and len(diag['episodes']) == 8,
            'Wrong diagnostic output')
    shapes = dict(grip_cmd=(800, 8), grasped=(800, 8), tcp=(800, 8, 3), parcel_xy=(800, 8, 6, 2),
                  parcel_z=(800, 8, 6), sorted_cnt=(800, 8))
    with np.load(npz_path, allow_pickle=False) as archive:
        require(len(archive.files) == len(shapes) and set(archive.files) == set(shapes), 'Wrong diagnostic trace keys')
        traces = {key: archive[key] for key in shapes}
    for key, shape in shapes.items():
        x = traces[key]
        require(x.shape == shape and x.dtype == (np.dtype(bool) if key == 'grasped' else np.dtype('float32'))
                and np.isfinite(x).all(), f'Invalid diagnostic trace shape/dtype/finite: {key}')
        require(not np.any(x[-1]), f'Unexpected diagnostic padding: {key}')  # shared logger executes 799 steps
    require(np.all(np.abs(traces['grip_cmd']) <= 1), 'Invalid diagnostic gripper range')
    sorted_cnt = traces['sorted_cnt']
    require(np.all((sorted_cnt >= 0) & (sorted_cnt <= 6) & (sorted_cnt == np.floor(sorted_cnt))),
            'Invalid diagnostic sorted range')
    for e, row in enumerate(diag['episodes']):
        require(row['env'] == e and row['seed'] == 5000 + e and row['parcels'] == 6, 'Wrong diagnostic scenes')
        numeric_range(row['sorted'], 0, 6, 'diagnostic sorted', integer=True)
        for key in ('first_close_step', 'first_grasp_step'):
            numeric_range(row[key], -1, 798, f'diagnostic {key}', integer=True)
        numeric_range(row['gripper_flips'], 0, 798, 'diagnostic flips', integer=True)
        numeric_range(row['longest_close_run'], 0, 799, 'diagnostic close run', integer=True)
        if row['n_plans'] is not None:
            numeric_range(row['n_plans'], 1, 799, 'diagnostic plans', integer=True)
        errors, reached = row['min_xy_err_per_parcel'], row['reached_within_2cm']
        require(len(errors) == len(reached) == 6 and all(type(x) is bool for x in reached), 'Wrong diagnostic parcel arrays')
        for error in errors:
            numeric_range(error, 0, math.inf, 'diagnostic xy error')
        closed = traces['grip_cmd'][:799, e] < 0
        grasped = traces['grasped'][:799, e]
        first_close = int(np.argmax(closed)) if closed.any() else -1
        first_grasp = int(np.argmax(grasped)) if grasped.any() else -1
        flips = int(np.sum(np.abs(np.diff(closed.astype(int)))))
        longest = run = 0
        for c in closed:
            run = run + 1 if c else 0
            longest = max(longest, run)
        distance = np.linalg.norm(traces['tcp'][:799, e, None, :2] - traces['parcel_xy'][:799, e], axis=-1)
        require(np.isfinite(distance).all(), 'Nonfinite diagnostic derived distance')
        minima = distance.min(axis=0)
        require(row['sorted'] == float(sorted_cnt[798, e]) and row['first_close_step'] == first_close
                and row['first_grasp_step'] == first_grasp and row['gripper_flips'] == flips
                and row['longest_close_run'] == longest, 'Diagnostic JSON/trace summary mismatch')
        require(np.allclose(errors, [round(float(x), 4) for x in minima], rtol=0, atol=1e-7)
                and reached == [bool(x < 0.02) for x in minima], 'Diagnostic JSON/trace distances mismatch')
        error = row['xy_err_at_first_close']
        if first_close == -1:
            require(error is None, 'Diagnostic no-close error must be null')
        else:
            numeric_range(error, 0, math.inf, 'diagnostic first-close error')
            require(math.isclose(error, float(distance[first_close].min()), rel_tol=0, abs_tol=1e-7),
                    'Diagnostic first-close trace mismatch')
    require(before == {p.name: sha(p) for p in (json_path, npz_path)}, 'Diagnostics changed during validation')
    require(sha(no_links(Path(checkpoint))) == digest, 'Diagnostic checkpoint changed')
    return dict(episodes=8, processed_steps=799, hashes=before, finite_json_and_traces=True,
                shapes={k: list(v) for k, v in shapes.items()})


def validate_completed_outputs(local):
    local = Path(local)
    summary = json.loads((local / 'run/summary.json').read_text())
    require(finite_tree(summary), 'Nonfinite final summary')
    selected, digest = child(local, summary['selected_checkpoint']), summary['checkpoint_sha256']
    results = []
    for h, d in SWEEP:
        rows = [json.loads(s) for s in (local / f'evaluations/h{h}_d{d}/metrics.jsonl').read_text().splitlines() if s.strip()]
        require(len(rows) == 1, 'Expected one saved sweep row')
        results.append(validate_eval(rows[0], selected, digest, 32, h, d))
    winner = max(results, key=lambda row: row['sort_accuracy'])
    require(results == summary['eval_runs'] == json.loads((local / 'run/evaluation_results.json').read_text())
            and winner == summary['winner'], 'Saved evaluation results/winner differ')
    opts = winner['policy_kwargs']
    rows = [json.loads(s) for s in (local / 'evaluations/preview/metrics.jsonl').read_text().splitlines() if s.strip()]
    require(len(rows) == 1, 'Expected one saved preview row')
    validate_eval(rows[0], selected, digest, 1, opts['act_horizon'], opts['num_inference_steps'])
    report = validate_diagnostics(local / 'evaluations/diagnostic', selected, digest)
    require(report == summary['diagnostics'], 'Saved diagnostics report differs')


def file_meta(path):
    st = Path(path).stat()
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)


def atomic_copy_verified(src, dest, guard):
    src, dest = no_links(src), no_links(dest)
    guard()
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = file_meta(src)
    digest = sha(src)
    tmp = dest.with_name(dest.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with src.open('rb') as source, tmp.open('xb') as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        require(file_meta(src) == before and sha(src) == digest and sha(tmp) == digest, f'Source changed during copy: {src}')
        guard()
        no_links(dest)
        os.replace(tmp, dest)
        guard()
        require(sha(dest) == digest, f'Destination hash mismatch: {dest}')
        return dict(meta=before, sha256=digest)
    finally:
        # Do not write/unlink in the underlying mount directory after a mount loss.
        try:
            guard()
        except Exception:
            pass
        else:
            tmp.unlink(missing_ok=True)


def mirror_files(local):
    priority = {'train/ckpts/latest.pt': 0, 'train/ckpts/final.pt': 1,
                'train/ckpts/best_eval_sort_accuracy.pt': 2}
    for p in sorted(Path(local).rglob('*'), key=lambda p: (priority.get(p.relative_to(local).as_posix(), 3), str(p))):
        no_links(p)
        rel = p.relative_to(local)
        if '.sync' in rel.parts or '.tmp.' in p.name or p.name.endswith(('.tmp', '.lock')):
            continue
        if rel.as_posix() == 'run/durability.json':
            continue
        if p.is_file():
            yield p


def sync_tree_once(local, remote, cache=None, *, final=False, guard):
    guard()
    cache = dict(cache or {})
    errors = []
    for src in mirror_files(local):
        rel = src.relative_to(local).as_posix()
        dest = child(remote, rel)
        if not final and cache.get(rel, {}).get('meta') == file_meta(src) and dest.is_file():
            continue
        try:
            # Recover progress from a previous timed-out worker without trusting destination size alone.
            meta = file_meta(src)
            if not final and dest.is_file() and dest.stat().st_size == meta['size']:
                digest = sha(src)
                if sha(dest) == digest and file_meta(src) == meta:
                    cache[rel] = dict(meta=meta, sha256=digest)
                    continue
            cache[rel] = atomic_copy_verified(src, dest, guard)
        except Exception as exc:
            errors.append(dict(path=rel, error=str(exc)))
    return dict(ok=not errors, cache=cache, errors=errors)


def required_artifacts(local):
    required = {'run/plan.json', 'run/source-install.json', 'run/summary.json', 'run/checks.json',
                'run/smoke.json', 'run/launch-hardware.json', 'run/launch-hardware.log', 'run/final-checkpoint.json', 'run/selection.json', 'run/evaluation_results.json',
                'run/train.log', 'run/smoke.log', 'run/checks.log', 'train/ckpts/final.pt', 'train/ckpts/latest.pt',
                'train/results.json', 'smoke/batch-audit.json', 'smoke/ckpts/final.pt',
                'evaluations/preview/metrics.jsonl', 'evaluations/diagnostic/diagnostics.json',
                'evaluations/diagnostic/diagnostics_traces.npz'}
    summary = json.loads((Path(local) / 'run/summary.json').read_text())
    require(summary['compute_status'] == 'completed', 'Compute has not completed')
    required.add(summary['selected_checkpoint'])
    required.update(summary['videos'])
    require(bool(summary['videos']), 'Missing required preview video')
    for h, d in SWEEP:
        required.add(f'evaluations/h{h}_d{d}/metrics.jsonl')
    plan = json.loads((Path(local) / 'run/plan.json').read_text())
    required.update('sources/' + rel for rel in plan['source_sha256'])
    return required


def finalize_durability(local, remote, result, guard):
    require(result.get('ok'), 'Cannot certify failed sync')
    guard()
    validate_completed_outputs(local)
    manifest = {}
    for src in mirror_files(local):
        rel = src.relative_to(local).as_posix()
        if rel == 'run/status.json':
            continue  # The receipt and live status are never part of the immutable manifest.
        entry = result['cache'].get(rel)
        require(entry and file_meta(src) == entry['meta'], f'Final source not stable: {rel}')
        require(sha(src) == entry['sha256'] == sha(child(remote, rel)), f'Final hash mismatch: {rel}')
        manifest[rel] = dict(sha256=entry['sha256'], size=entry['meta']['size'])
    require(required_artifacts(local) <= manifest.keys(), 'Required final artifacts missing')
    guard()
    receipt = dict(remote_verified=True, verified_at=time.time(), files=manifest, remote_root=str(remote))
    receipt_path = Path(local) / 'run/durability.json'
    write_json(receipt_path, receipt)
    atomic_copy_verified(receipt_path, child(remote, 'run/durability.json'), guard)
    return receipt


def sync_worker(plan_path, request, response):
    plan = load_plan(plan_path)
    local, remote = roots(plan['exp'])
    guard = lambda: (roots(plan['exp']), ensure_drive())
    req = json.loads(child(local, request).read_text())
    result = sync_tree_once(local, remote, req.get('cache'), final=req['final'], guard=guard)
    if req['final'] and result['ok']:
        try:
            finalize_durability(local, remote, result, guard)
            status = local / 'run/status.json'
            state = json.loads(status.read_text())
            state.update(status='completed', remote_status='verified', remote_verified=True)
            write_json(status, state)
            atomic_copy_verified(status, child(remote, 'run/status.json'), guard)
        except Exception as exc:
            result['ok'] = False
            result['errors'].append(dict(path='run/durability.json', error=str(exc)))
    write_json(child(local, response), result)


def bounded_sync(plan, cache, final=False):
    local = Path(plan['local_root'])
    token = uuid.uuid4().hex
    request, response = f'run/.sync/{token}.request.json', f'run/.sync/{token}.response.json'
    try:
        write_json(child(local, request), dict(cache=cache, final=final))
        cmd = [plan['python'], str(Path(plan['repo']) / 'tools/run_colab_hard.py'),
               str(local / 'run/plan.json'), '--sync-worker', request, response]
        # run(timeout) kills/waits only this own worker; it has no children.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FINAL_SYNC_TIMEOUT if final else SYNC_TIMEOUT)
        require(result.returncode == 0, 'Sync worker failed: ' + result.stderr[-1500:])
        return json.loads(child(local, response).read_text())
    except Exception as exc:
        return dict(ok=False, cache=cache, errors=[dict(path='.', error=str(exc))])
    finally:
        child(local, request).unlink(missing_ok=True)
        child(local, response).unlink(missing_ok=True)


def owned_group_live(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    if sys.platform.startswith('linux'):
        # An unreaped zombie cannot execute or hold FDs. Colab's init may retain such
        # entries after their dispatcher exits; distinguish them from live descendants.
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue  # exited while enumerating
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        return False
    # Other POSIX hosts conservatively wait until the whole group disappears.
    return True


def stop_owned_process(p, grace=15, kill_grace=5):
    # p.pid is the process group created by start_new_session. Never target a discovered PID/group.
    require(p.pid != os.getpgrp(), 'Refusing to signal supervisor group')
    saved = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            saved[sig] = signal.signal(sig, signal.SIG_IGN)
    try:
        for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, kill_grace)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + duration
            while True:
                p.poll()  # reap the leader independently of surviving descendants
                if not owned_group_live(p.pid) and p.poll() is not None:
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        raise RuntimeError(f'Owned process group {p.pid} still live after SIGKILL')
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def run_stage(cmd, log, timeout, update, env, cwd, sync=None, lock_fds=()):
    reject_gpu_processes()
    if lock_fds:
        args = cmd[1:]
        if args[0] == '-u':
            args = args[1:]
        entry = Path(args[0])
        entry = entry.relative_to(cwd).as_posix() if entry.is_absolute() else entry.as_posix()
        cmd = [cmd[0], '-u', str(Path(cwd) / 'tools/hard_gpu_worker.py'), '--repo', str(cwd),
               '--local', env['MARSO_HARD_LOCAL_ROOT'], '--lock-fds', *map(str, lock_fds), entry, *args[1:]]
    log = no_links(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream, ThreadPoolExecutor(max_workers=1) as sync_pool:
        pending_sync = None
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True, pass_fds=lock_fds)
        started = time.monotonic()
        next_progress, next_sync = 0.0, 0.0
        try:
            update(stage_pid=p.pid, command=cmd, log=str(log))
            while p.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    raise TimeoutError(f'Stage deadline {timeout}s: {log}')
                if now >= next_progress:
                    with log.open('rb') as tail:
                        tail.seek(max(0, log.stat().st_size - 1500))
                        progress = tail.read().decode(errors='replace').replace('\r', '\n').splitlines()
                    latest = next((x for x in reversed(progress) if x.strip()), 'initializing')
                    update(elapsed_s=round(now - started), latest_progress=latest[-350:])
                    print(f'[hard] pid={p.pid} elapsed={now-started:.0f}s {latest[-350:]}', flush=True)
                    next_progress = now + PROGRESS_SECONDS
                if pending_sync is not None and pending_sync.done():
                    try:
                        pending_sync.result()
                    except Exception as exc:
                        print(f'[hard] periodic sync pending: {exc}', flush=True)
                    pending_sync = None
                if sync and pending_sync is None and now >= next_sync:
                    pending_sync = sync_pool.submit(sync)
                    next_sync = now + PROGRESS_SECONDS
                time.sleep(0.1)
            require(p.returncode == 0, f'Stage exit {p.returncode}; see {log}')
            require(not owned_group_live(p.pid), 'Stage leader exited with live descendants')
        except BaseException:
            stop_owned_process(p)
            raise
        finally:
            update(stage_pid=None)


def verified_checks(plan):
    local = Path(plan['local_root'])
    checks = json.loads((local / 'run/checks.json').read_text())
    require(checks['source_sha256'] == plan['source_sha256'], 'Stale checks sources')
    for name, digest in checks['dataset']['hashes'].items():
        require(sha(child(Path(plan['repo']) / 'il/demos/hard', name)) == digest, 'Dataset changed after checks')
    require(checks['dataset']['h5_trajectories'] == 200 and checks['reference']['reference_valid']
            and checks['reference']['num_parcels'] == 6 and checks['reference']['max_episode_steps'] == 800
            and checks['hardware']['device'] == 'cuda', 'Invalid checks evidence')
    return checks


def preset_probe(repo):
    from omegaconf import OmegaConf
    difficulty = OmegaConf.to_container(OmegaConf.load(Path(repo) / 'conf/difficulty/hard.yaml'), resolve=True)
    method = OmegaConf.to_container(OmegaConf.load(Path(repo) / 'il/conf/method/dp_rgb_hard.yaml'), resolve=True)
    require(difficulty['difficulty'] == dict(name='hard', num_parcels=6, fixed_poses=False, max_episode_steps=800),
            'Hard difficulty source contract changed')
    require(difficulty['randomization'] == RANDOMIZATION, 'Hard randomization source contract changed')
    for key, value in dict(baseline_dir='diffusion_policy', script='train_rgbd.py', demo_kind='rgb',
                           demo_dir='hard', max_episode_steps=800).items():
        require(method[key] == value, f'Hard method source contract changed: {key}')
    for key, value in PRESET.items():
        if key not in ('seed', 'save_freq'):
            require(key in method['flags'] and method['flags'][key] == value, f'Hard preset changed: {key}')
    return dict(difficulty=difficulty, method=method, launch_overrides=dict(seed=1, save_freq=1000))


def choose_checkpoint(local, final_report):
    final = Path(local) / 'train/ckpts/final.pt'
    best = Path(local) / 'train/ckpts/best_eval_sort_accuracy.pt'
    fallback = not best.exists()
    selected = final if fallback else best
    selected_report = checkpoint_report(selected, final=fallback)
    best_score = max(row['sort_accuracy'] for row in final_report['evaluation_history'])
    require(not fallback or best_score == 0, 'Positive best metric but best checkpoint missing')
    require(selected_report['best_sort_accuracy'] == best_score, 'Selected checkpoint does not preserve best metric')
    if not fallback:
        require(best_score > 0, 'Strict-improvement trainer cannot produce a zero-score best')
        history = final_report['evaluation_history']
        first = next(i for i, row in enumerate(history) if row['sort_accuracy'] == best_score)
        require(selected_report['iteration'] == history[first]['iteration'], 'Best checkpoint is not the first maximum iteration')
        require(selected_report['evaluation_history'] == history[:first + 1], 'Best checkpoint evaluation history differs')
    return selected, selected_report, fallback


def perform_checks(plan):
    return dict(source_sha256=plan['source_sha256'], presets=preset_probe(plan['repo']), hardware=gpu_probe(),
                dataset=dataset_probe(Path(plan['repo']) / DEMO), reference=reference_probe(plan['repo']))


def require_phase_ready(state, phase):
    if phase == 'finalize':
        require(state.get('compute_status') == 'completed', 'Only completed compute can retry durability')
    else:
        expected = {'checks': 'prepared', 'smoke': 'checks_passed', 'run': 'smoke_passed'}
        require(phase in expected and state.get('status') == expected[phase],
                f'Phase {phase} requires {expected.get(phase)}; refusing relaunch/reuse')


def pipeline(plan, phase):
    local = Path(plan['local_root'])
    status_path = child(local, 'run/status.json')
    state = json.loads(status_path.read_text())
    cache = {}
    status_lock = threading.Lock()

    def update(**kw):
        with status_lock:
            state.update(kw, updated=time.time())
            write_json(status_path, state)

    def sync(final=False):
        nonlocal cache
        result = bounded_sync(plan, cache, final)
        cache = result.get('cache', cache)
        update(remote_status='verified' if final and result['ok'] else ('synced' if result['ok'] else 'pending'),
               remote_verified=bool(final and result['ok']), sync_errors=result.get('errors', []))
        return result

    env = dict(os.environ, DISPLAY='', PYOPENGL_PLATFORM='egl', HDF5_USE_FILE_LOCKING='FALSE',
               PYTHONUNBUFFERED='1', PYTHONPATH=plan['repo'], MARSO_HARD_LOCAL_ROOT=str(local),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with launch_locks(Path(plan['repo']), local) as lock_fds:
        stage = partial(run_stage, lock_fds=lock_fds)
        verify_sources(Path(plan['repo']), plan['source_sha256'])
        state = json.loads(status_path.read_text())
        require_phase_ready(state, phase)  # Rejections leave previous evidence/status untouched.
        try:
            if phase == 'checks':
                require(state['status'] == 'prepared', 'Checks require a fresh prepared run')
                update(status='checking')
                stage([plan['python'], str(Path(plan['repo']) / 'tools/run_colab_hard.py'),
                           str(local / 'run/plan.json'), '--probe', '--lock-fds', *map(str, lock_fds)], local / 'run/checks.log', 1800,
                          update, env, plan['repo'], sync)
                verified_checks(plan)
                update(status='checks_passed')
            elif phase == 'smoke':
                require(state['status'] == 'checks_passed', 'Smoke requires actual checks')
                verified_checks(plan)
                (local / 'smoke').mkdir(exist_ok=False)
                update(status='smoke_training')
                stage(train_command(plan, smoke=True), local / 'run/smoke.log', 1800, update, env, plan['repo'], sync)
                report = checkpoint_report(local / 'smoke/ckpts/final.pt', 100)
                audit_path = local / 'smoke/batch-audit.json'
                audit = json.loads(audit_path.read_text())
                report.update(validate_smoke_audit(audit, report))
                report.update(batch_audit_sha256=sha(audit_path),
                              checks_sha256=sha(local / 'run/checks.json'), source_sha256=plan['source_sha256'], seeds_full_training=False)
                write_json(local / 'run/smoke.json', report)
                update(status='smoke_passed')
            elif phase == 'run':
                require(state['status'] == 'smoke_passed', 'Full training requires completed smoke; no relaunch/reuse')
                verified_checks(plan)
                smoke = json.loads((local / 'run/smoke.json').read_text())
                require(smoke['checks_sha256'] == sha(local / 'run/checks.json') and smoke['source_sha256'] == plan['source_sha256'], 'Stale smoke evidence')
                require(smoke['sha256'] == sha(local / 'smoke/ckpts/final.pt'), 'Smoke checkpoint changed')
                smoke_report = checkpoint_report(local / 'smoke/ckpts/final.pt', 100)
                require(smoke['batch_audit_sha256'] == sha(local / 'smoke/batch-audit.json'), 'Smoke batch audit changed')
                validate_smoke_audit(json.loads((local / 'smoke/batch-audit.json').read_text()), smoke_report)
                # Refresh CUDA evidence immediately before full launch. No smoke weights are loaded by the trainer.
                stage([plan['python'], str(Path(plan['repo']) / 'tools/run_colab_hard.py'),
                       str(local / 'run/plan.json'), '--hardware-probe', '--lock-fds', *map(str, lock_fds)],
                      local / 'run/launch-hardware.log', 180, update, env, plan['repo'], sync)
                (local / 'train').mkdir(exist_ok=False)
                update(status='training', compute_status='running')
                stage(train_command(plan), local / 'run/train.log', TRAIN_TIMEOUT, update, env, plan['repo'], sync)
                final = local / 'train/ckpts/final.pt'
                final_report = checkpoint_report(final)
                write_json(local / 'run/final-checkpoint.json', final_report)
                selected, selected_report, fallback = choose_checkpoint(local, final_report)
                digest = selected_report['sha256']
                write_json(local / 'run/selection.json', dict(**selected_report, fallback_to_final=fallback,
                           reason='No positive best sort metric checkpoint was written' if fallback else 'Preserved training-time best sort accuracy checkpoint'))
                results = []
                for h, d in SWEEP:
                    label = f'h{h}_d{d}'
                    cmd, folder = eval_command(plan, selected, label, 32, h, d)
                    update(status='sweeping', evaluation=label)
                    stage(cmd, folder / 'eval.log', 3600, update, env, plan['repo'], sync)
                    rows = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines() if x.strip()]
                    require(len(rows) == 1, 'Expected one fresh evaluation row')
                    results.append(validate_eval(rows[0], selected, digest, 32, h, d))
                    write_json(local / 'run/evaluation_results.json', results)
                winner = max(results, key=lambda row: row['sort_accuracy'])
                opts = winner['policy_kwargs']
                h, d = opts['act_horizon'], opts['num_inference_steps']
                cmd, folder = eval_command(plan, selected, 'preview', 1, h, d, video=True)
                update(status='preview')
                stage(cmd, folder / 'eval.log', 1800, update, env, plan['repo'], sync)
                rows = [json.loads(x) for x in (folder / 'metrics.jsonl').read_text().splitlines() if x.strip()]
                require(len(rows) == 1, 'Missing preview metrics')
                validate_eval(rows[0], selected, digest, 1, h, d)
                videos = [p.relative_to(local).as_posix() for p in (folder / 'videos').rglob('*.mp4') if p.stat().st_size > 0]
                require(videos, 'No preview MP4 produced')
                cmd, folder = eval_command(plan, selected, 'diagnostic', 8, h, d, diagnostic=True)
                update(status='diagnostics')
                stage(cmd, folder / 'eval.log', 1800, update, env, plan['repo'], sync)
                diagnostic_report = validate_diagnostics(folder, selected, digest)
                require(sha(selected) == digest, 'Selected checkpoint changed')
                verify_sources(Path(plan['repo']), plan['source_sha256'])
                verified_checks(plan)
                write_json(local / 'run/summary.json', dict(compute_status='completed', exp=plan['exp'],
                           iterations=40000, level='hard', selected_checkpoint=selected.relative_to(local).as_posix(),
                           checkpoint_sha256=digest, fallback_to_final=fallback, winner=winner, eval_runs=results,
                           videos=videos, diagnostics=diagnostic_report, diagnostics_steps=799, evaluator_steps=799, configured_max_episode_steps=800,
                           score_scope='local validation seeds; not official or heldout', durability_receipt='run/durability.json'))
                update(status='sync_pending', compute_status='completed')
                outcome = sync(final=True)
                update(status='completed' if outcome['ok'] else 'sync_pending')
            elif phase == 'finalize':
                require(state['compute_status'] == 'completed', 'Only completed compute can retry durability')
                outcome = sync(final=True)
                update(status='completed' if outcome['ok'] else 'sync_pending')
            else:
                raise ValueError('Unknown phase')
            if phase in ('checks', 'smoke'):
                sync()
        except BaseException as exc:
            update(status='failed', error=f'{type(exc).__name__}: {exc}')
            sync()
            raise
    print(json.dumps(state, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--phase', choices=['checks', 'smoke', 'run', 'finalize'])
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--hardware-probe', action='store_true')
    parser.add_argument('--lock-fds', nargs=3, type=int, default=())
    parser.add_argument('--sync-worker', nargs=2)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.sync_worker:
        sync_worker(args.plan, *args.sync_worker)
    elif args.hardware_probe:
        verify_inherited_locks(args.lock_fds, plan['repo'], plan['local_root'])
        write_json(Path(plan['local_root']) / 'run/launch-hardware.json', gpu_probe())
    elif args.probe:
        verify_inherited_locks(args.lock_fds, plan['repo'], plan['local_root'])
        write_json(Path(plan['local_root']) / 'run/checks.json', perform_checks(plan))
    else:
        require(args.phase is not None, '--phase is required')
        pipeline(plan, args.phase)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
