"""Hard-only regression tests; no CUDA, installs, external data, or Drive required."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

from tools import run_colab_hard as hard


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    original = hard.subprocess.check_output
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda cmd, **kw: '' if cmd[0] == 'ps' else original(cmd, **kw))
    root = tmp_path.resolve()
    scope = dict(repo=str(root / 'repo'), python=sys.executable, drive=str(root / 'drive'),
                 mount='/', local=str(root / 'local'))
    Path(scope['repo']).mkdir()
    Path(scope['drive']).mkdir()
    bundle = {}
    for rel in hard.SOURCE_FILES:
        payload = b'# reviewed fixture\n'
        bundle[rel] = dict(base_sha256=None, sha256=hashlib.sha256(payload).hexdigest(),
                           content_b64=base64.b64encode(payload).decode())
    return scope, bundle


def test_bootstrap_new_roots_source_reuse_backups_and_plan(sandbox):
    scope, bundle = sandbox
    existing = Path(scope['repo']) / 'eval.py'
    existing.write_text('# base\n')
    bundle['eval.py']['base_sha256'] = hard.sha(existing)
    plan_path = hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    plan = hard.load_plan(plan_path, scope)
    local, remote = hard.roots(plan['exp'], scope)
    assert (local / 'source-backups/eval.py').read_text() == '# base\n'
    assert (local / 'sources/eval.py').read_bytes() == existing.read_bytes()
    assert remote.is_dir()
    assert not (Path(scope['drive']) / 'demos/hard').exists()  # marker deliberately absent
    with pytest.raises(ValueError, match='reuse'):
        hard.bootstrap(bundle, plan['exp'], scope)
    existing.write_text('# unexpected\n')
    with pytest.raises(ValueError, match='Source changed'):
        hard.load_plan(plan_path, scope)


@pytest.mark.parametrize('bad', ['hard_../evil', '../hard_20260914_010203', 'medium_20260914_010203',
                                  'hard_20260914_010203/x', 'hard_x', 'hard_２０２６０９１４_010203'])
def test_scope_rejection(sandbox, bad):
    scope, _ = sandbox
    with pytest.raises(ValueError):
        hard.roots(bad, scope)


@pytest.mark.parametrize('reuse_remote', [False, True])
def test_even_empty_roots_are_rejected(sandbox, reuse_remote):
    scope, bundle = sandbox
    roots = hard.roots('hard_20260914_010203', scope)
    roots[int(reuse_remote)].mkdir(parents=True)
    with pytest.raises(ValueError, match='reuse'):
        hard.bootstrap(bundle, 'hard_20260914_010203', scope)


def test_unknown_runtime_source_is_rejected_before_writes(sandbox):
    scope, bundle = sandbox
    (Path(scope['repo']) / 'eval.py').write_text('unexpected patch')
    with pytest.raises(ValueError, match='Unexpected runtime source'):
        hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    assert not Path(scope['local']).exists()
    assert not (Path(scope['drive']) / 'recoveries').exists()


def test_symlink_and_escape_rejection(sandbox):
    scope, bundle = sandbox
    (Path(scope['repo']) / 'eval.py').symlink_to(Path(scope['drive']) / 'unrelated')
    with pytest.raises(ValueError, match='Symlink'):
        hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    for rel in ('../outside', '/absolute', 'tools/../../outside'):
        with pytest.raises(ValueError, match='Unsafe'):
            hard.child(Path(scope['repo']), rel)


def test_mount_required_even_with_existing_root(sandbox):
    scope, _ = sandbox
    scope['mount'] = str(Path(scope['drive']).parent)
    with pytest.raises(ValueError, match='Real Drive mount'):
        hard.ensure_drive(scope)


def create_dataset(root, n=200):
    path = root / 'trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5'
    meta = dict(env_info=dict(env_id='WarehouseSort-v1', env_kwargs=dict(num_parcels=6, fixed_poses=False,
                obs_mode='rgb', obs_camera='scene', control_mode='pd_ee_delta_pos', max_episode_steps=800,
                randomization=copy.deepcopy(hard.RANDOMIZATION))), episodes=[{}] * n)
    path.with_suffix('.json').write_text(json.dumps(meta))
    with h5py.File(path, 'w') as f:
        for i in range(n):
            g = f.create_group(f'traj_{i}')
            g['actions'] = np.zeros((1, 4), np.float32)
            for name, dim in [('obs/agent/qpos', 9), ('obs/agent/qvel', 9), ('obs/extra/tcp_pose', 7)]:
                g[name] = np.zeros((2, dim), np.float32)
            g['obs/extra/is_grasped'] = np.zeros((2,), np.float32)
            g.create_dataset('obs/sensor_data/scene_camera/rgb', shape=(2, 128, 128, 3), dtype='u1', compression='gzip')
    return path


def test_dataset_all_200_and_schema_hashes(tmp_path):
    path = create_dataset(tmp_path.resolve())
    result = hard.dataset_probe(path)
    assert result['h5_trajectories'] == 200 and result['state_dim'] == 26 and result['finite_state']
    assert result['hashes'][path.name] == hard.sha(path)
    with h5py.File(path, 'a') as f:
        f['traj_17/obs/extra/is_grasped'][0] = np.nan
    with pytest.raises(ValueError, match='grasp'):
        hard.dataset_probe(path)


@pytest.mark.parametrize('kind', ['count', 'parcels', 'pose', 'bin', 'rgb', 'state', 'actions'])
def test_dataset_bad_contract(tmp_path, kind):
    path = create_dataset(tmp_path.resolve(), 199 if kind == 'count' else 200)
    if kind in ('parcels', 'pose', 'bin'):
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        kw = meta['env_info']['env_kwargs']
        if kind == 'parcels':
            kw['num_parcels'] = 4
        elif kind == 'pose':
            kw['randomization']['parcel_pose']['xy_jitter'] = [0, 0]
        else:
            kw['randomization']['bin_position']['side_swap_prob'] = 0
        meta_path.write_text(json.dumps(meta))
    elif kind in ('rgb', 'state', 'actions'):
        with h5py.File(path, 'a') as f:
            if kind == 'rgb':
                del f['traj_0/obs/sensor_data/scene_camera/rgb']
                f['traj_0/obs/sensor_data/scene_camera/rgb'] = np.zeros((2, 64, 64, 3), np.uint8)
            elif kind == 'state':
                f['traj_199/obs/agent/qpos'][1, 3] = np.inf
            else:
                f['traj_1/actions'][0, 0] = np.nan
    with pytest.raises(ValueError):
        hard.dataset_probe(path)


def checkpoint(total=40000, iteration=None):
    iteration = total if iteration is None else iteration
    steps = total if iteration == total else iteration + 1
    args = dict(hard.PRESET, total_iters=total, max_episode_steps=800, num_parcels=2)
    if total == 100:
        args['eval_freq'] = 0
    cfg = dict(args, state_dim=26, act_dim=4, image_hw=[128, 128], demo_paths=[str(Path(hard.RUNTIME['repo']) / hard.DEMO)])
    return dict(iteration=iteration, args=args, config=cfg, agent={'weight': torch.ones(2)},
                ema_agent={'weight': torch.ones(2)}, optimizer={'state': {0: {'step': torch.tensor(steps),
                'exp_avg': torch.ones(2)}}, 'param_groups': [{'lr': 0.0}]},
                lr_scheduler={'last_epoch': steps}, ema_state={'optimization_step': steps, 'shadow_params': [torch.ones(2)]},
                scaler={'scale': 65536.0}, rng={'python': (3,), 'numpy': ('MT19937',), 'torch': torch.ones(4, dtype=torch.uint8),
                'cuda': [torch.ones(4, dtype=torch.uint8)]}, best_eval_metrics={'sort_accuracy': 0.0},
                eval_history=[] if total == 100 else [dict(iteration=i, sort_accuracy=0.0, n_episodes=32) for i in range(5000, iteration + 1, 5000)])


def test_final_scheduler_actual_loop_and_legacy_parcels():
    ck = checkpoint()
    assert ck['args']['num_parcels'] == 2
    assert hard.validate_checkpoint(ck)['scheduler_step'] == 40000
    ck['lr_scheduler']['last_epoch'] = 40001
    with pytest.raises(ValueError, match='Scheduler'):
        hard.validate_checkpoint(ck)
    assert hard.validate_checkpoint(checkpoint(iteration=35000), final=False)['scheduler_step'] == 35001


@pytest.mark.parametrize('section', ['agent', 'ema_agent', 'optimizer', 'ema_state', 'rng', 'scaler'])
def test_checkpoint_missing_state(section):
    ck = checkpoint()
    del ck[section]
    with pytest.raises(ValueError, match='Missing durable'):
        hard.validate_checkpoint(ck)


@pytest.mark.parametrize('section', ['agent', 'ema_agent', 'optimizer', 'ema_state'])
def test_checkpoint_nonfinite_state(section):
    ck = checkpoint()
    if section == 'optimizer':
        ck[section]['state'][0]['exp_avg'][0] = torch.nan
    elif section == 'ema_state':
        ck[section]['shadow_params'][0][0] = torch.inf
    else:
        ck[section]['weight'][0] = torch.nan
    with pytest.raises(ValueError):
        hard.validate_checkpoint(ck)


@pytest.mark.parametrize('updates', [100, 98, 1])
def test_smoke_records_positive_actual_updates_separately(updates):
    ck = checkpoint(100)
    ck['optimizer']['state'][0]['step'] = torch.tensor(updates)
    report = hard.validate_checkpoint(ck, 100)
    assert report['processed_batches'] == report['scheduler_step'] == 100
    assert report['successful_optimizer_updates'] == updates
    assert report['skipped_optimizer_updates'] == 100 - updates


@pytest.mark.parametrize('updates', [0, -1, 101, float('nan'), float('inf'), 98.5])
def test_smoke_rejects_invalid_optimizer_counts(updates):
    ck = checkpoint(100)
    ck['optimizer']['state'][0]['step'] = torch.tensor(updates)
    with pytest.raises(ValueError):
        hard.validate_checkpoint(ck, 100)


def test_commands_are_fresh_hard_local_seeded_and_bounded(tmp_path):
    plan = dict(repo=hard.RUNTIME['repo'], python=hard.RUNTIME['python'], local_root=str(tmp_path.resolve()))
    full = hard.train_command(plan)
    smoke = hard.train_command(plan, smoke=True)
    assert 'method=dp_rgb_hard' in full and 'flags.total_iters=40000' in full
    assert 'flags.save_freq=1000' in full and 'flags.num_demos=null' in full and '+flags.seed=1' in full
    assert 'flags.resume=null' in full and not any('smoke' in c for c in full)
    assert 'flags.total_iters=100' in smoke and 'flags.eval_freq=0' in smoke
    assert hard.TRAIN_TIMEOUT >= 4 * 3600 and hard.PROGRESS_SECONDS == 60
    assert all('act/' not in c.lower() for c in full)
    for h, d in hard.SWEEP:
        cmd, folder = hard.eval_command(plan, Path('/checkpoint.pt'), f'h{h}_d{d}', 32, h, d)
        assert 'difficulty=hard' in cmd and 'num_envs=8' in cmd and 'record_video=false' in cmd
        assert f'+policy_kwargs.act_horizon={h}' in cmd and f'+policy_kwargs.num_inference_steps={d}' in cmd
        assert '5031' in (folder / 'seeds.yaml').read_text()
    cmd, _ = hard.eval_command(plan, Path('/checkpoint.pt'), 'preview', 1, 4, 32, video=True)
    assert 'num_envs=1' in cmd and 'video_envs=1' in cmd
    cmd, _ = hard.eval_command(plan, Path('/checkpoint.pt'), 'diag', 8, 4, 32, diagnostic=True)
    assert 'tools/rollout_logger.py' in cmd and '+log_episodes=8' in cmd


def test_eval_requires_actual_contract_and_checkpoint_hash(tmp_path):
    path = tmp_path.resolve() / 'best.pt'
    path.write_bytes(b'best')
    row = eval_row(path)
    digest = hard.sha(path)
    assert hard.validate_eval(row, path, digest, 32, 8, 16)['checkpoint_sha256'] == digest
    for key, value in [('level', 'medium'), ('max_episode_steps', 500), ('n_episodes', 8), ('num_parcels', 4)]:
        with pytest.raises(ValueError):
            hard.validate_eval(dict(row, **{key: value}), path, digest, 32, 8, 16)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        hard.validate_eval(row, path, digest, 32, 8, 16)


def test_atomic_source_changing_same_size_and_mtime(tmp_path, monkeypatch):
    src, dest = tmp_path.resolve() / 'src', tmp_path.resolve() / 'dest'
    src.write_bytes(b'old')
    dest.write_bytes(b'previous')
    old = src.stat()
    original = hard.shutil.copyfileobj
    def changing(source, target, length):
        original(source, target, length)
        src.write_bytes(b'new')
        os.utime(src, ns=(old.st_atime_ns, old.st_mtime_ns))
    monkeypatch.setattr(hard.shutil, 'copyfileobj', changing)
    with pytest.raises(ValueError, match='Source changed'):
        hard.atomic_copy_verified(src, dest, lambda: None)
    assert dest.read_bytes() == b'previous'
    assert not list(tmp_path.glob('*.tmp.*'))


def test_atomic_mount_loss_never_promotes_copy(tmp_path):
    src, dest = tmp_path.resolve() / 'src', tmp_path.resolve() / 'remote/dest'
    src.write_bytes(b'payload')
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError('mount lost')
    with pytest.raises(RuntimeError, match='mount lost'):
        hard.atomic_copy_verified(src, dest, guard)
    assert not dest.exists()


def finished_tree(tmp_path):
    local, remote = tmp_path.resolve() / 'local', tmp_path.resolve() / 'remote'
    local.mkdir()
    remote.mkdir()
    hard.write_json(local / 'run/summary.json', dict(compute_status='completed', selected_checkpoint='train/ckpts/best_eval_sort_accuracy.pt',
                                                   videos=['evaluations/preview/videos/one.mp4']))
    hard.write_json(local / 'run/plan.json', dict(source_sha256={'eval.py': 'test'}))
    for rel in hard.required_artifacts(local):
        path = hard.child(local, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b'immutable artifact')
    selected = local / 'train/ckpts/best_eval_sort_accuracy.pt'
    results = []
    for h, d in hard.SWEEP:
        row = eval_row(selected, h=h, d=d)
        (local / f'evaluations/h{h}_d{d}/metrics.jsonl').write_text(json.dumps(row) + '\n')
        results.append(hard.validate_eval(row, selected, hard.sha(selected), 32, h, d))
    (local / 'evaluations/preview/metrics.jsonl').write_text(json.dumps(eval_row(selected, n=1)) + '\n')
    hard.write_json(local / 'run/evaluation_results.json', results)
    write_diagnostics(local / 'evaluations/diagnostic', selected)
    diagnostic = hard.validate_diagnostics(local / 'evaluations/diagnostic', selected, hard.sha(selected))
    hard.write_json(local / 'run/summary.json', dict(compute_status='completed',
                    selected_checkpoint=selected.relative_to(local).as_posix(), checkpoint_sha256=hard.sha(selected),
                    videos=['evaluations/preview/videos/one.mp4'], eval_runs=results, winner=results[0], diagnostics=diagnostic))
    hard.write_json(local / 'run/status.json', dict(status='sync_pending', compute_status='completed'))
    return local, remote


def test_receipt_immutable_files_and_exclusions(tmp_path):
    local, remote = finished_tree(tmp_path)
    result = hard.sync_tree_once(local, remote, final=True, guard=lambda: None)
    receipt = hard.finalize_durability(local, remote, result, lambda: None)
    assert 'train/ckpts/best_eval_sort_accuracy.pt' in receipt['files']
    assert 'run/status.json' not in receipt['files'] and 'run/durability.json' not in receipt['files']
    assert hard.sha(local / 'run/durability.json') == hard.sha(remote / 'run/durability.json')
    hard.write_json(local / 'run/status.json', dict(status='completed'))
    hard.finalize_durability(local, remote, result, lambda: None)  # mutable status is allowed to change


@pytest.mark.parametrize('change', ['local', 'remote', 'missing_required', 'failed_sync'])
def test_receipt_rejects_invalid_final_state(tmp_path, change):
    local, remote = finished_tree(tmp_path)
    result = hard.sync_tree_once(local, remote, final=True, guard=lambda: None)
    rel = 'train/ckpts/final.pt'
    if change == 'local':
        path = local / rel
        old = path.stat()
        path.write_bytes(b'x' * old.st_size)
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif change == 'remote':
        (remote / rel).write_bytes(b'corrupt')
    elif change == 'missing_required':
        (local / rel).unlink()
    else:
        result['ok'] = False
    with pytest.raises(ValueError):
        hard.finalize_durability(local, remote, result, lambda: None)
    assert not (local / 'run/durability.json').exists()


def test_sync_rejects_nested_symlink(tmp_path):
    local, remote = tmp_path.resolve() / 'local', tmp_path.resolve() / 'remote'
    local.mkdir()
    remote.mkdir()
    (local / 'linked').symlink_to(remote, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        hard.sync_tree_once(local, remote, guard=lambda: None)


def test_bounded_worker_timeout_is_nonfatal_pending(tmp_path, monkeypatch):
    import subprocess
    def timeout(*args, **kwargs):
        assert kwargs['timeout'] == hard.SYNC_TIMEOUT
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    result = hard.bounded_sync(dict(local_root=str(tmp_path.resolve()), repo='/repo', python=sys.executable), {})
    assert not result['ok'] and result['errors']
    assert not list((tmp_path / 'run/.sync').iterdir())


def test_duplicate_detection_and_lock_contention(tmp_path, monkeypatch):
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda *a, **k: '123 python il/train.py method=dp_rgb_hard\n')
    with pytest.raises(RuntimeError, match='already active'):
        hard.reject_gpu_processes()
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda *a, **k: '')
    repo, local = tmp_path.resolve() / 'repo', tmp_path.resolve() / 'local'
    repo.mkdir(); local.mkdir()
    with hard.launch_locks(repo, local) as fds:
        hard.verify_inherited_locks(fds, repo, local)
        with pytest.raises(RuntimeError, match='lock busy'):
            with hard.launch_locks(repo, local):
                pass
    with pytest.raises(ValueError, match='inherited'):
        hard.verify_inherited_locks((), repo, local)


def test_stage_timeout_only_stops_own_group(tmp_path, monkeypatch):
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda *a, **k: '')
    import subprocess
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True)
    try:
        with pytest.raises(TimeoutError):
            hard.run_stage([sys.executable, '-c', 'import time; time.sleep(10)'],
                           tmp_path.resolve() / 'stage.log', 0.05, lambda **kw: None, os.environ.copy(), str(tmp_path))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_real_hydra_composition_matches_hard_contract(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    repo = Path(__file__).resolve().parents[1]
    report = hard.preset_probe(repo)
    assert report['difficulty']['difficulty']['num_parcels'] == 6
    plan = dict(python=sys.executable, local_root=str(tmp_path.resolve()))
    for smoke in (False, True):
        with initialize_config_dir(config_dir=str(repo / 'il/conf'), version_base=None):
            cfg = compose(config_name='train', overrides=hard.train_command(plan, smoke=smoke)[3:])
        assert cfg.demo_dir == 'hard' and cfg.max_episode_steps == 800 and cfg.baseline_dir == 'diffusion_policy'
        assert cfg.flags.total_iters == (100 if smoke else 40000)
        assert cfg.flags.batch_size == 128 and cfg.flags.lr == 1e-4
        assert cfg.flags.seed == 1 and cfg.flags.amp and cfg.flags.num_demos is None and cfg.flags.resume is None
        assert cfg.flags.save_freq == 1000
        assert Path(cfg.flags.exp_name).is_relative_to(tmp_path.resolve())


def test_best_preserved_and_zero_only_fallback(tmp_path):
    root = tmp_path.resolve()
    ckdir = root / 'train/ckpts'
    ckdir.mkdir(parents=True)
    final = checkpoint()
    torch.save(final, ckdir / 'final.pt')
    report = hard.checkpoint_report(ckdir / 'final.pt')
    selected, _, fallback = hard.choose_checkpoint(root, report)
    assert selected.name == 'final.pt' and fallback
    report['evaluation_history'][0]['sort_accuracy'] = 0.5
    with pytest.raises(ValueError, match='Positive best'):
        hard.choose_checkpoint(root, report)
    best = checkpoint(iteration=5000)
    best['best_eval_metrics']['sort_accuracy'] = 0.5
    best['eval_history'] = copy.deepcopy(report['evaluation_history'][:1])
    torch.save(best, ckdir / 'best_eval_sort_accuracy.pt')
    selected, _, fallback = hard.choose_checkpoint(root, report)
    assert selected.name == 'best_eval_sort_accuracy.pt' and not fallback
    assert hard.checkpoint_report(ckdir / 'final.pt')['best_sort_accuracy'] == 0.0


def test_real_sampler_scheduler_final_boundary():
    from diffusion_policy.utils import IterationBasedBatchSampler
    from torch.utils.data import BatchSampler, SequentialSampler
    sampler = IterationBasedBatchSampler(BatchSampler(SequentialSampler(range(4)), 2, False), 40000, start_iter=39997)
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
    state = scheduler.state_dict()
    state['last_epoch'] = 39997
    scheduler.load_state_dict(state)
    for iteration, _ in enumerate(sampler, start=39997):
        optimizer.zero_grad()
        parameter.square().backward()
        optimizer.step()
        scheduler.step()
    assert iteration == 39999 and scheduler.last_epoch == 40000
    # Assert that the checked-in trainer still assigns the final label from total_iters.
    import ast
    source = Path(__file__).resolve().parents[1] / 'il/baselines/diffusion_policy/train_rgbd.py'
    assignments = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'final_iter' for t in n.targets)]
    assert len(assignments) == 1 and ast.unparse(assignments[0].value) == 'args.total_iters'


def test_retry_sync_rehashes_existing_destination_without_recopy(tmp_path, monkeypatch):
    local, remote = tmp_path.resolve() / 'local', tmp_path.resolve() / 'remote'
    local.mkdir(); remote.mkdir()
    (local / 'payload').write_bytes(b'existing verified copy')
    hard.sync_tree_once(local, remote, guard=lambda: None)
    def no_recopy(*args, **kwargs):
        raise AssertionError('Should reconstruct cache from actual hashes')
    monkeypatch.setattr(hard, 'atomic_copy_verified', no_recopy)
    result = hard.sync_tree_once(local, remote, cache={}, guard=lambda: None)
    assert result['ok'] and 'payload' in result['cache']


def test_sync_prioritizes_latest_final_best(tmp_path):
    local = tmp_path.resolve()
    for rel in ('aaa.log', 'train/ckpts/best_eval_sort_accuracy.pt', 'train/ckpts/final.pt', 'train/ckpts/latest.pt'):
        p = hard.child(local, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'test')
    assert [p.name for p in hard.mirror_files(local)][:3] == ['latest.pt', 'final.pt', 'best_eval_sort_accuracy.pt']


def test_periodic_sync_exception_does_not_fail_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda *a, **k: '')
    def unavailable():
        raise RuntimeError('Drive unavailable')
    events = []
    hard.run_stage([sys.executable, '-c', 'import time; time.sleep(0.05)'],
                   tmp_path.resolve() / 'stage.log', 5, lambda **kw: events.append(kw),
                   os.environ.copy(), str(tmp_path), sync=unavailable)
    assert events[-1]['stage_pid'] is None


def test_stage_rechecks_duplicate_before_creating_log(tmp_path, monkeypatch):
    monkeypatch.setattr(hard.subprocess, 'check_output', lambda *a, **k: '54321 python eval.py difficulty=medium\n')
    path = tmp_path.resolve() / 'not-started.log'
    with pytest.raises(RuntimeError, match='already active'):
        hard.run_stage([sys.executable, '-c', 'raise AssertionError("must not launch")'], path, 5,
                       lambda **kw: None, os.environ.copy(), str(tmp_path))
    assert not path.exists()


def test_relaunch_rejection_does_not_rewrite_prior_status(sandbox):
    scope, bundle = sandbox
    path = hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    plan = hard.load_plan(path, scope)
    status = path.parent / 'status.json'
    before = status.read_bytes()
    with pytest.raises(ValueError, match='refusing relaunch'):
        hard.pipeline(plan, 'run')
    assert status.read_bytes() == before
    assert not (path.parent / 'train.log').exists()
    completed = dict(status='completed', compute_status='completed', remote_verified=True)
    hard.write_json(status, completed)
    before = status.read_bytes()
    with pytest.raises(ValueError, match='refusing relaunch'):
        hard.pipeline(plan, 'checks')
    assert status.read_bytes() == before


def eval_row(path, n=32, h=8, d=16):
    return dict(level='hard', obs_mode='rgb', n_episodes=n, requested_n_episodes=n, num_parcels=6,
                max_episode_steps=800, seed0=5000, checkpoint=str(path), sort_accuracy=0.0,
                all_placed_rate=0.0, mis_sort_rate=0.0, mean_sorted=0.0, mean_steps=799.0, eval_seconds=1.0,
                policy_kwargs=dict(act_horizon=h, num_inference_steps=d))


def write_diagnostics(folder, checkpoint_path):
    folder.mkdir(parents=True, exist_ok=True)
    traces = dict(grip_cmd=np.zeros((800, 8), np.float32), grasped=np.zeros((800, 8), bool),
                  tcp=np.zeros((800, 8, 3), np.float32), parcel_xy=np.zeros((800, 8, 6, 2), np.float32),
                  parcel_z=np.zeros((800, 8, 6), np.float32), sorted_cnt=np.zeros((800, 8), np.float32))
    episodes = [dict(env=e, seed=5000+e, sorted=0.0, parcels=6, first_close_step=-1, first_grasp_step=-1,
                     gripper_flips=0, longest_close_run=0, xy_err_at_first_close=None,
                     min_xy_err_per_parcel=[0.0]*6, reached_within_2cm=[True]*6, n_plans=100) for e in range(8)]
    hard.write_json(folder / 'diagnostics.json', dict(level='hard', checkpoint=str(checkpoint_path), episodes=episodes))
    np.savez_compressed(folder / 'diagnostics_traces.npz', **traces)
    return traces


@pytest.mark.parametrize('field', ['sort_accuracy', 'eval_seconds', 'mean_sorted', 'all_placed_rate',
                                  'mean_steps', 'mis_sort_rate', 'extra'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_all_eval_fields_must_be_finite(tmp_path, field, value):
    path = tmp_path.resolve() / 'checkpoint.pt'
    path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError, match='Nonfinite'):
        hard.validate_eval(dict(eval_row(path), **{field: value}), path, hard.sha(path), 32, 8, 16)


@pytest.mark.parametrize('field,value', [('mean_steps', 801), ('eval_seconds', -1), ('mean_sorted', 7),
                                       ('sort_accuracy', 1.1), ('mis_sort_rate', -0.1), ('all_placed_rate', True)])
def test_eval_range_contract(tmp_path, field, value):
    path = tmp_path.resolve() / 'checkpoint.pt'
    path.write_bytes(b'checkpoint')
    with pytest.raises(ValueError):
        hard.validate_eval(dict(eval_row(path), **{field: value}), path, hard.sha(path), 32, 8, 16)


@pytest.mark.parametrize('field,value', [('sorted', float('nan')), ('xy_err_at_first_close', float('inf')),
                                       ('sorted', 7), ('first_close_step', 799), ('gripper_flips', -1),
                                       ('min_xy_err_per_parcel', [float('nan')]*6),
                                       ('min_xy_err_per_parcel', [0]*5), ('min_xy_err_per_parcel', [1]*6)])
def test_durability_rejects_corrupt_diagnostic_json_even_when_hashes_match(tmp_path, field, value):
    local, remote = finished_tree(tmp_path)
    path = local / 'evaluations/diagnostic/diagnostics.json'
    data = json.loads(path.read_text())
    data['episodes'][0][field] = value
    path.write_text(json.dumps(data))  # simulate malformed runtime output, including nonstandard JSON numbers
    result = hard.sync_tree_once(local, remote, final=True, guard=lambda: None)
    assert result['ok'] and hard.sha(path) == hard.sha(remote / path.relative_to(local))
    with pytest.raises(ValueError):
        hard.finalize_durability(local, remote, result, lambda: None)
    assert not (local / 'run/durability.json').exists()


@pytest.mark.parametrize('field', ['grip_cmd', 'tcp', 'parcel_xy', 'parcel_z', 'sorted_cnt', 'grasped'])
@pytest.mark.parametrize('change', ['nonfinite', 'shape', 'range_or_type'])
def test_diagnostic_trace_validation(tmp_path, field, change):
    folder = tmp_path.resolve() / 'diagnostic'
    selected = tmp_path.resolve() / 'selected.pt'
    selected.write_bytes(b'checkpoint')
    traces = write_diagnostics(folder, selected)
    if change == 'shape':
        traces[field] = traces[field][:-1]
    elif change == 'nonfinite':
        traces[field] = traces[field].astype(np.float32)
        traces[field].flat[0] = np.inf
    elif field in ('grip_cmd', 'sorted_cnt'):
        traces[field].flat[0] = 7.5
    else:
        traces[field] = traces[field].astype(np.float64)
    np.savez_compressed(folder / 'diagnostics_traces.npz', **traces)
    with pytest.raises(ValueError):
        hard.validate_diagnostics(folder, selected, hard.sha(selected))


def test_durability_rejects_nan_npz_even_with_matching_hashes(tmp_path):
    local, remote = finished_tree(tmp_path)
    path = local / 'evaluations/diagnostic/diagnostics_traces.npz'
    with np.load(path) as archive:
        traces = {k: archive[k] for k in archive.files}
    traces['tcp'][0, 0, 0] = np.nan
    np.savez_compressed(path, **traces)
    result = hard.sync_tree_once(local, remote, final=True, guard=lambda: None)
    with pytest.raises(ValueError, match='finite'):
        hard.finalize_durability(local, remote, result, lambda: None)


@pytest.mark.parametrize('mutation', ['wrong_iteration', 'empty_history', 'different_history', 'later_tie'])
def test_best_requires_exact_first_maximum_checkpoint(tmp_path, mutation):
    folder = tmp_path.resolve() / 'train/ckpts'
    folder.mkdir(parents=True)
    final = checkpoint()
    final['eval_history'][0]['sort_accuracy'] = 0.5
    final['eval_history'][1]['sort_accuracy'] = 0.5  # later tie must not replace strict-improvement best
    final['best_eval_metrics']['sort_accuracy'] = 0.5
    torch.save(final, folder / 'final.pt')
    best = checkpoint(iteration=10000 if mutation == 'later_tie' else 123 if mutation == 'wrong_iteration' else 5000)
    best['best_eval_metrics']['sort_accuracy'] = 0.5
    best['eval_history'] = copy.deepcopy(final['eval_history'][:2 if mutation == 'later_tie' else 1])
    if mutation == 'wrong_iteration' or mutation == 'empty_history':
        best['eval_history'] = []
    if mutation == 'different_history':
        best['eval_history'][0]['sort_accuracy'] = 0.4
    torch.save(best, folder / 'best_eval_sort_accuracy.pt')
    with pytest.raises(ValueError):
        hard.choose_checkpoint(tmp_path.resolve(), hard.checkpoint_report(folder / 'final.pt'))


@pytest.mark.parametrize('rel', ['warehouse_sort/il_policy.py', 'il/baselines/diffusion_policy/train_rgbd.py',
                               'il/conf/method/dp_rgb_hard.yaml', 'conf/difficulty/hard.yaml'])
def test_manifest_omissions_rejected_before_writes(sandbox, rel):
    scope, bundle = sandbox
    del bundle[rel]
    with pytest.raises(ValueError, match='source set'):
        hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    assert not Path(scope['local']).exists()
    with pytest.raises(ValueError, match='source set'):
        hard.verify_sources(Path(scope['repo']), {k: v['sha256'] for k,v in bundle.items()})


@pytest.mark.parametrize('rel', ['tools/unreviewed.py', 'unexpected.txt', '../outside', 'warehouse_sort/extra.py'])
def test_manifest_extra_payload_rejected(sandbox, rel):
    scope, bundle = sandbox
    bundle[rel] = bundle['eval.py']
    with pytest.raises(ValueError, match='source set'):
        hard.bootstrap(bundle, 'hard_20260914_010203', scope)
    assert not Path(scope['local']).exists()


def test_real_cpu_amp_initial_scale_recovery_keeps_batch_scheduler_ema_contract():
    from tools.hard_gpu_worker import SmokeAudit
    from diffusers.training_utils import EMAModel
    torch.manual_seed(1)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    ema = EMAModel(model.parameters(), power=0.75)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536.0)
    with SmokeAudit() as audit:
        for _ in range(100):
            optimizer.zero_grad()
            with torch.autocast('cpu', dtype=torch.float16):
                loss = model(torch.ones(1, 1)).square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.step(model.parameters())
    ck = checkpoint(100)
    ck.update(agent=model.state_dict(), ema_agent=model.state_dict(), optimizer=optimizer.state_dict(),
              lr_scheduler=scheduler.state_dict(), ema_state=ema.state_dict(), scaler=scaler.state_dict())
    report = hard.validate_checkpoint(ck, 100)
    assert report['processed_batches'] == scheduler.last_epoch == ema.optimization_step == 100
    assert report['successful_optimizer_updates'] == 98
    assert report['skipped_optimizer_updates'] == 2 and model.weight.item() < 1.0
    assert hard.validate_smoke_audit(audit.report(), report, device='cpu')['loss_count'] == 100
    bad = audit.report()
    bad['losses'][50] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        hard.validate_smoke_audit(bad, report, device='cpu')


@pytest.mark.parametrize('field,value', [('processed_batches', 99), ('successful_optimizer_updates', 0),
                                       ('skipped_optimizer_updates', 100), ('losses', [1.0]*99),
                                       ('devices', ['cpu']), ('successful_update_batches', [1]*98)])
def test_smoke_audit_requires_real_complete_evidence(field, value):
    ck = checkpoint(100)
    ck['optimizer']['state'][0]['step'] = torch.tensor(98)
    audit = dict(processed_batches=100, successful_optimizer_updates=98, skipped_optimizer_updates=2,
                 successful_update_batches=list(range(3, 101)), losses=[1.0]*100, devices=['cuda'],
                 finite_successful_gradients_and_state=True)
    audit[field] = value
    with pytest.raises(ValueError):
        hard.validate_smoke_audit(audit, hard.validate_checkpoint(ck, 100))


def reference_limit_fixture(limit=800, with_limit=True):
    import gymnasium as gym
    from omegaconf import OmegaConf

    class Base(gym.Env):
        spec = None
        max_episode_steps = 100  # legacy constructor metadata, not the active truncation limit
        observation_space = gym.spaces.Discrete(1)
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return 0, {}

        def step(self, action):
            return 0, 0.0, False, False, {}

    base = Base()
    env = gym.wrappers.OrderEnforcing(base)
    if with_limit:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=limit)
    env = gym.Wrapper(env)  # a real outer wrapper must delegate the attribute lookup
    cfg = OmegaConf.create(dict(max_episode_steps=800, difficulty=dict(max_episode_steps=800)))
    return env, cfg


def test_reference_time_limit_real_wrapper_chain_without_spec():
    env, cfg = reference_limit_fixture()
    try:
        assert env.spec is None and env.env.spec is None and env.unwrapped.spec is None
        report = hard.reference_time_limit(env, cfg)
        assert report == dict(max_episode_steps=800, wrapper_max_episode_steps=800,
                              configured_max_episode_steps=800, difficulty_max_episode_steps=800,
                              base_max_episode_steps=100, spec_max_episode_steps=None)
        env.reset()
        for _ in range(799):
            assert env.step(0)[3] is False
        assert env.step(0)[3] is True  # real Gym TimeLimit truncates at the verified budget
        assert env.unwrapped.max_episode_steps == 100  # verifier does not bind/mutate task metadata
    finally:
        env.close()


@pytest.mark.parametrize('limit', [100, 799, 801, None])
def test_reference_time_limit_rejects_wrong_actual_limit_despite_valid_config(limit):
    env, cfg = reference_limit_fixture()
    env.env._max_episode_steps = limit
    try:
        with pytest.raises(ValueError, match='Actual wrapper step limit != 800'):
            hard.reference_time_limit(env, cfg)
    finally:
        env.close()


def test_reference_time_limit_rejects_missing_wrapper_despite_800_metadata():
    from gymnasium.envs.registration import EnvSpec
    env, cfg = reference_limit_fixture(with_limit=False)
    env.unwrapped.spec = EnvSpec('CPUFixture-v0', max_episode_steps=800)
    env.unwrapped.max_episode_steps = 800
    try:
        with pytest.raises(ValueError, match='Actual wrapper step limit missing'):
            hard.reference_time_limit(env, cfg)
    finally:
        env.close()


@pytest.mark.parametrize('config_key', ['max_episode_steps', 'difficulty.max_episode_steps'])
def test_reference_time_limit_still_rejects_wrong_configuration(config_key):
    from omegaconf import OmegaConf
    env, cfg = reference_limit_fixture()
    OmegaConf.update(cfg, config_key, 100)
    try:
        with pytest.raises(ValueError, match='Configured Hard step limit'):
            hard.reference_time_limit(env, cfg)
    finally:
        env.close()
