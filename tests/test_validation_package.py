"""CPU-only contracts; no training, simulator, installs, or real model inference."""
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import select
import sys
import types

import pytest
import torch

from tools import validation_common as common
from tools import run_generalization as general
from tools import build_repro_package as package


def full_manifest(tmp_path):
    value = json.loads((common.ROOT / 'docs/VALIDATION_INPUT.example.json').read_text())
    value['checkpoints']['hard'].update(path='hard_best.pt', sha256='a' * 64, act_horizon=8,
                                      num_inference_steps=16, training_completed_iterations=40000)
    path = tmp_path / 'inputs.json'
    path.write_text(json.dumps(value))
    return path, value


def test_protocol_freeze_is_complete_and_disjoint():
    frozen = general.frozen_protocols()
    for protocol, start in [('fresh_seed', 6000), ('stress', 8000)]:
        for level in common.LEVELS:
            spec = frozen['protocols'][protocol][level]
            assert spec['seeds'] == list(range(start, start + 100))
            assert spec['n_episodes'] == 100 and spec['num_envs'] == 4
            assert spec['max_episode_steps'] == common.BUDGETS[level]
            assert spec['rollout_step_calls'] == common.BUDGETS[level] - 1
            assert spec['num_parcels'] == common.PARCELS[level]
            assert spec['inference_noise_seed'] == 0
    assert frozen['protocols']['fresh_seed']['easy']['fixed_poses'] is True
    assert frozen['protocols']['stress']['easy']['fixed_poses'] is False
    assert frozen['protocols']['stress']['easy']['randomization']['parcel_pose']['xy_jitter'] == [-.01, .01]
    assert frozen['protocols']['stress']['medium']['randomization']['parcel_pose']['xy_jitter'] == [-.025, .025]
    assert frozen['protocols']['stress']['hard']['randomization']['parcel_pose']['yaw_jitter'] == [-.15, .15]


def test_dry_plans_leave_hard_pending_and_do_not_create_output(tmp_path):
    for script in ('run_generalization.py', 'build_repro_package.py'):
        target = tmp_path / script
        r = subprocess.run([sys.executable, str(common.ROOT / 'tools' / script), '--dry-run', '--out', str(target)],
                           capture_output=True, text=True, timeout=15, check=True)
        plan = json.loads(r.stdout)
        assert plan['inputs']['checkpoints']['hard']['sha256'] is None
        assert 'warehouse_sort/utils.py' in plan['source_sha256']
        assert not target.exists()
    with pytest.raises(ValueError, match='pending'):
        common.load_manifest(common.ROOT / 'docs/VALIDATION_INPUT.example.json')


@pytest.mark.parametrize('field,value', [('sha256', 'b' * 64), ('act_horizon', 4), ('num_inference_steps', 8)])
def test_reviewed_winners_cannot_change(tmp_path, field, value):
    path, data = full_manifest(tmp_path)
    data['checkpoints']['easy'][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='reviewed winner'):
        common.load_manifest(path)


def test_manifest_requires_all_levels_budget_and_stable_label(tmp_path):
    path, data = full_manifest(tmp_path)
    original = copy.deepcopy(data)
    for mutation in ('missing', 'budget', 'label'):
        data = copy.deepcopy(original)
        if mutation == 'missing':
            del data['checkpoints']['hard']
        elif mutation == 'budget':
            data['checkpoints']['hard']['training_completed_iterations'] = 30000
        else:
            data['checkpoints']['hard']['provenance_label'] = '/Users/private/model.pt'
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            common.load_manifest(path)


def rows_for(spec):
    return [dict(seed=s, num_parcels=spec['num_parcels'], max_episode_steps=spec['max_episode_steps'],
                 step_calls=spec['rollout_step_calls'], sorted=1., mis_sorted=0., all_placed=False,
                 steps_to_complete=spec['max_episode_steps']) for s in spec['seeds']]


@pytest.mark.parametrize('bad', ['count', 'duplicate', 'nan', 'range', 'budget', 'aggregate'])
def test_result_contract_rejects_invalid_outputs(bad):
    spec = general.frozen_protocols()['protocols']['fresh_seed']['easy']
    rows = rows_for(spec)
    metrics = general.aggregate_rows(rows, spec)
    if bad == 'count': rows.pop()
    if bad == 'duplicate': rows[-1]['seed'] = rows[0]['seed']
    if bad == 'nan': rows[0]['sorted'] = float('nan')
    if bad == 'range': rows[0]['mis_sorted'] = 3
    if bad == 'budget': rows[0]['max_episode_steps'] = 999
    if bad == 'aggregate': metrics['sort_accuracy'] = .9
    with pytest.raises(ValueError):
        general.validate_metrics(metrics, rows, spec)


def test_weighting_never_renormalizes_missing_levels():
    values = {l: {'sort_accuracy': x} for l, x in zip(common.LEVELS, (1., .5, .2))}
    assert general.weighted_score(values) == pytest.approx(.45)
    del values['hard']
    with pytest.raises(ValueError, match='all three'):
        general.weighted_score(values)


def actual_shared_rollout():
    # Compile the real shared helper; avoid importing unavailable simulator packages.
    tree = ast.parse((common.ROOT / 'warehouse_sort/utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('rollout_metrics', 'expand_seeds', 'to_device')]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'warehouse_sort/utils.py', 'exec'), namespace)
    return namespace['rollout_metrics']


def test_shared_observers_record_once_and_preserve_padded_batch_metrics():
    class Env:
        num_envs, num_parcels = 4, 2
        def __init__(self):
            self.unwrapped = self
            self.resets, self.steps, self.evaluations = [], 0, 0
        def reset(self, seed):
            self.resets.append(seed)
            return torch.zeros(4, 1), {}
        def step(self, action):
            self.steps += 1
            return torch.zeros(4, 1), None, None, None, {}
        def evaluate(self):
            self.evaluations += 1
            return {'success_count': torch.tensor([0., 1., 2., 1.]), 'mis_sort_count': torch.zeros(4),
                    'all_placed': torch.tensor([False, False, True, False]), 'steps_to_complete': torch.ones(4) * 3}
    class Agent:
        def reset(self): pass
        def act(self, obs, deterministic=True): return torch.zeros(4, 4)
    plain, observed = Env(), Env()
    reset_seeds, outcome_seeds = [], []
    fn = actual_shared_rollout()
    expected = fn(plain, Agent(), 'cpu', 6, list(range(6000, 6006)), 5)
    actual = fn(observed, Agent(), 'cpu', 6, list(range(6000, 6006)), 5,
                on_reset=lambda base, obs, seeds: reset_seeds.extend(seeds),
                on_batch=lambda base, ev, seeds: outcome_seeds.extend(seeds))
    assert actual == expected
    assert reset_seeds == outcome_seeds == list(range(6000, 6006))
    assert observed.steps == plain.steps == 8
    assert observed.evaluations == 2 and len(observed.resets) == 2
    assert observed.resets[-1] == [6004, 6005, 6000, 6001]


def pose_rows(spec, jitter=True):
    rows = []
    for i, seed in enumerate(spec['seeds']):
        dx = (.003 if i % 2 else -.003) if jitter else 0
        theta = (.02 if i % 2 else -.02) if jitter else 0
        import math
        rows.append({'seed': seed, 'fixed_poses': spec['fixed_poses'],
                     'nominal': {'inbound_center': [0, 0], 'parcel_half': [.02, .02, .02], 'bin_base_x': .4, 'bin_base_y': .3},
                     'parcel_poses': [[x + dx, dx, .021, math.cos(theta / 2), 0, 0, math.sin(theta / 2)] for x in (-.05, .05)],
                     'bin_poses': [[.4 + dx, sign * (.3 + dx), 0, 1, 0, 0, 0] for sign in (1, -1)]})
    return rows


def test_easy_stress_checks_actual_fixed_override_and_jitter():
    spec = general.frozen_protocols()['protocols']['stress']['easy']
    rows = pose_rows(spec)
    assert general.verify_pose_effect(rows, spec)['n_resets'] == 100
    rows[0]['fixed_poses'] = True
    with pytest.raises(ValueError, match='override ignored'):
        general.verify_pose_effect(rows, spec)
    with pytest.raises(ValueError, match='no observed effect'):
        general.verify_pose_effect(pose_rows(spec, jitter=False), spec)
    rows = pose_rows(spec)
    rows[0]['parcel_poses'][0][0] += .1
    with pytest.raises(ValueError, match='outside frozen range'):
        general.verify_pose_effect(rows, spec)


def test_checked_policy_rejects_privileged_or_bad_actions():
    obs = {'rgb': torch.zeros(2, 8, 8, 3, dtype=torch.uint8), 'state': torch.zeros(2, 26)}
    agent = types.SimpleNamespace(act=lambda obs, deterministic: torch.zeros(2, 4))
    checked = general.CheckedPolicy(agent)
    assert checked.act(obs).shape == (2, 4)
    with pytest.raises(ValueError, match='proprio only'):
        checked.act(dict(obs, parcel_poses=torch.zeros(2, 6)))
    for action in (torch.zeros(2, 3), torch.full((2, 4), float('nan')), torch.full((2, 4), 2.)):
        agent.act = lambda obs, deterministic, a=action: a
        with pytest.raises(ValueError): checked.act(obs)


def checkpoint(level='hard'):
    return {'ema_agent': {'weight': torch.tensor([0., -0., 1., 2.]), 'counter': torch.tensor(4)},
            'agent': {'weight': torch.tensor([99.])}, 'iteration': 25000,
            'config': {'obs_mode': 'rgb', 'state_dim': 26, 'act_dim': 4, 'max_episode_steps': common.BUDGETS[level],
                       'demo_paths': [f'/Users/private/demo/{level}/trajectory.h5'], 'total_iters': common.COMPLETED[level],
                       'obs_horizon': 2, 'pred_horizon': 16, 'act_horizon': 8, 'num_diffusion_iters': 100,
                       'num_inference_steps': 16, 'image_hw': [128, 128, 3]},
            'args': {'num_parcels': 2}, 'optimizer': {'irrelevant': 'not packaged'}}


def entry_for(path, level='hard'):
    return {'path': str(path), 'sha256': common.sha256(path), 'provenance_label': f'{level}-verified-best',
            'selection': 'best_eval_sort_accuracy', 'act_horizon': 8, 'num_inference_steps': 32,
            'training_completed_iterations': common.COMPLETED[level]}


@pytest.mark.parametrize('weight_key', ['ema_agent', 'model'])
def test_compact_ema_exact_roundtrip_and_sanitized_source(tmp_path, weight_key):
    ckpt = checkpoint()
    if weight_key == 'model': ckpt['model'] = ckpt.pop('ema_agent')
    source, dest = tmp_path / 'source.pt', tmp_path / 'compact.pt'
    torch.save(ckpt, source)
    provenance = package.compact_checkpoint(entry_for(source), 'hard', dest)
    compact = torch.load(dest, weights_only=False)
    assert set(compact) == {'model', 'config', 'iteration', 'source'}
    assert compact['source'] == 'hard-verified-best'
    assert '/Users/' not in json.dumps(compact['config'])
    assert compact['config']['act_horizon'] == 8 and compact['config']['num_inference_steps'] == 32
    assert provenance['source_weight_key'] == weight_key and provenance['exact_tensor_identity']
    package.assert_tensor_identity(ckpt[weight_key], compact['model'])
    assert provenance['compact_sha256'] == common.sha256(dest)


@pytest.mark.parametrize('bad', ['level', 'maxsteps', 'budget', 'nan', 'fp16', 'raw', 'horizon'])
def test_checkpoint_rejects_wrong_lineage_and_unsafe_conversions(tmp_path, bad):
    ckpt = checkpoint()
    entry = {'act_horizon': 8, 'num_inference_steps': 32}
    if bad == 'level': ckpt['config']['demo_paths'] = ['/content/demos/medium/demo.h5']
    if bad == 'maxsteps': ckpt['config']['max_episode_steps'] = 500
    if bad == 'budget': ckpt['config']['total_iters'] = 50000
    if bad == 'nan': ckpt['ema_agent']['weight'][0] = float('nan')
    if bad == 'fp16': ckpt['ema_agent']['weight'] = ckpt['ema_agent']['weight'].half()
    if bad == 'raw': del ckpt['ema_agent']
    if bad == 'horizon': entry['act_horizon'] = 16
    with pytest.raises(ValueError): common.checkpoint_contract(ckpt, 'hard', entry)


def test_signed_zero_identity_is_byte_exact():
    with pytest.raises(ValueError, match='tensor changed'):
        package.assert_tensor_identity({'a': torch.tensor([0.])}, {'a': torch.tensor([-0.])})


def test_new_directories_never_reuse_and_drive_is_rejected(tmp_path):
    a = common.new_output(tmp_path, 'validation_')
    b = common.new_output(tmp_path, 'validation_')
    assert a != b and a.is_dir() and b.is_dir()
    with pytest.raises(ValueError, match='local'):
        common.new_output(tmp_path / 'drive' / 'MyDrive', 'validation_')


def test_gpu_lock_excludes_other_process_and_retains_inode(tmp_path):
    lock = tmp_path / 'gpu.lock'
    with common.gpu_lock(lock):
        inode = lock.stat().st_ino
        result = subprocess.run([sys.executable, '-c',
            'import fcntl,sys; f=open(sys.argv[1],"a+"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)', str(lock)],
            capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and 'BlockingIOError' in result.stderr
    with common.gpu_lock(lock): assert lock.stat().st_ino == inode


def test_gpu_process_checks_fail_closed(monkeypatch):
    def command(argv, **kwargs):
        text = '42 /content/python train_rgbd.py --total-iters 40000\n' if argv[0] == 'ps' else ''
        return types.SimpleNamespace(stdout=text)
    monkeypatch.setattr(common.subprocess, 'run', command)
    with pytest.raises(ValueError, match='processes active'): common.check_idle_gpu()
    monkeypatch.setattr(common.subprocess, 'run', lambda argv, **kw: types.SimpleNamespace(stdout='' if argv[0] == 'ps' else '71\n'))
    with pytest.raises(ValueError, match='compute clients'): common.check_idle_gpu()


def test_supervisor_bounds_and_stops_owned_worker(tmp_path):
    with pytest.raises(TimeoutError):
        common.supervised([sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path, .15)
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['status'] == 'failed'
    with pytest.raises(ProcessLookupError): os.kill(progress['pid'], 0)


@pytest.mark.parametrize('name', ['../secret', '/tmp/secret', '.env', 'auth/token.json', 'demos/easy/a.json', 'outputs/raw.json'])
def test_allowlist_excludes_private_paths(name):
    with pytest.raises(ValueError): common.safe_relative(name)


def test_source_allowlist_includes_reviewed_import_dependencies():
    names = common.source_files()
    assert {'eval.py', 'tools/rollout_logger.py', 'warehouse_sort/il_policy.py',
            'il/baselines/diffusion_policy/train_rgbd.py',
            'il/baselines/diffusion_policy/diffusion_policy/evaluate.py'} <= set(names)
    assert not any('/demos/' in p or 'run_colab' in p or p.endswith('.ipynb') for p in names)
    assert not {'docs/EASY_MEDIUM_VERIFIED_INPUTS.json', 'package_worker_result.txt'} & set(names)
    assert not any(p.startswith('tests/') for p in names)


def test_archive_seal_rejects_tampering_and_is_deterministic(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    (root / 'README.md').write_text('release\n')
    common.seal_files(root)
    assert common.verify_seal(root) == 1
    package.archive_release(root, tmp_path / 'a.tar.gz')
    os.utime(root / 'README.md', (500, 500))
    package.archive_release(root, tmp_path / 'b.tar.gz')
    assert common.sha256(tmp_path / 'a.tar.gz') == common.sha256(tmp_path / 'b.tar.gz')
    (root / 'README.md').write_text('changed\n')
    with pytest.raises(ValueError, match='SHA mismatch'): common.verify_seal(root)


def test_build_clean_archive_from_synthetic_ema_and_explicit_artifacts(tmp_path, monkeypatch):
    plan = {'input_manifest_sha256': 'a' * 64, 'source_sha256': common.source_hashes(),
            'inputs': {'checkpoints': {}, 'artifacts': []}}
    for level in common.LEVELS:
        path = tmp_path / f'{level}.pt'
        torch.save(checkpoint(level), path)
        plan['inputs']['checkpoints'][level] = entry_for(path, level)
    artifact = tmp_path / 'summary.json'
    artifact.write_text('{"status":"synthetic-test-only"}\n')
    plan['inputs']['artifacts'] = [{'path': str(artifact), 'sha256': common.sha256(artifact), 'name': 'summary.json'}]
    monkeypatch.setattr(package, 'versions', lambda: {'python': 'test-only', 'packages': {'torch': {'version': 'test-only', 'license': 'test'}}})
    out = tmp_path / 'build'
    out.mkdir()
    plan_path = out / 'plan.json'
    common.write_json(plan_path, plan)
    package.build_worker(plan_path, out)
    result = json.loads((out / 'build_result.json').read_text())
    assert result['clean_extract_sha'] == 'passed' and result['clean_extract_gpu_smoke'] == 'pending'
    release = out / 'release'
    assert common.verify_seal(release) > 30
    assert (release / 'artifacts/summary.json').read_bytes() == artifact.read_bytes()
    sub = json.loads((release / 'submission.yaml').read_text())
    assert sub['rgb']['policy'] == common.ENTRYPOINT
    assert all(set(v) == {'checkpoint'} for v in sub['rgb']['levels'].values())
    assert '/Users/' not in (release / 'MODEL_PROVENANCE.json').read_text()
    assert str(tmp_path) not in (release / 'PACKAGE_INPUT.json').read_text()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop(common.GPU_LOCK_FD_ENV, None)
    r = subprocess.run([sys.executable, str(release / 'tools/build_repro_package.py'), '--verify-only'],
                       cwd=release, env=env, capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(r.stdout)['sha_verified_files'] > 30
    smoke_out = tmp_path / 'forbidden-smoke'
    r = subprocess.run([sys.executable, '-B', str(release / 'tools/build_repro_package.py'),
                        '--worker', 'smoke', str(release), str(smoke_out)], cwd=release, env=env,
                       capture_output=True, text=True, timeout=10)
    assert r.returncode != 0 and 'inherited GPU lock FD required' in r.stderr
    assert not smoke_out.exists()


def test_actual_constructor_and_wrapper_budget_are_both_bound():
    spec = general.frozen_protocols()['protocols']['fresh_seed']['hard']
    base = types.SimpleNamespace(max_episode_steps=100)
    wrapper = types.SimpleNamespace(get_wrapper_attr=lambda key: 800)
    env = types.SimpleNamespace(unwrapped=base, _env=wrapper)
    evidence = general.bind_actual_budget(env, spec)
    assert base.max_episode_steps == 800
    assert evidence == {'wrapper_max_episode_steps': 800, 'task_budget_before_binding': 100, 'task_max_episode_steps': 800}
    wrapper.get_wrapper_attr = lambda key: 100
    with pytest.raises(ValueError, match='TimeLimit'):
        general.bind_actual_budget(env, spec)


def test_easy_stress_override_reaches_actual_gym_make_arguments(monkeypatch):
    from omegaconf import OmegaConf
    tree = ast.parse((common.ROOT / 'warehouse_sort/utils.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('compose_cfg', '_gym_make')]
    captured = {}
    def gym_make(name, **kwargs):
        captured.update(kwargs)
    namespace = {'gym': types.SimpleNamespace(make=gym_make), 'OmegaConf': OmegaConf}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'warehouse_sort/utils.py', 'exec'), namespace)
    module = types.ModuleType('warehouse_sort.utils')
    module.compose_cfg = namespace['compose_cfg']
    monkeypatch.setitem(sys.modules, 'warehouse_sort.utils', module)
    spec = general.frozen_protocols()['protocols']['stress']['easy']
    cfg = general.resolved_config('easy', spec, '/content/easy.pt')
    namespace['_gym_make'](cfg, 'rgb', cfg.randomization, 4, None)
    assert captured['fixed_poses'] is False
    assert captured['randomization'] == spec['randomization']
    assert captured['max_episode_steps'] == 250 and captured['num_parcels'] == 2


def test_actual_cli_rejects_pending_hard_before_gpu_or_output(tmp_path):
    for script in ('run_generalization.py', 'build_repro_package.py'):
        out = tmp_path / script
        r = subprocess.run([sys.executable, str(common.ROOT / 'tools' / script), '--run', '--out', str(out)],
                           capture_output=True, text=True, timeout=15)
        assert r.returncode != 0 and 'pending checkpoint or winner settings' in r.stderr
        assert not out.exists()


def test_source_sha_mismatch_fails_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / 'bad.pt'
    path.write_bytes(b'not a checkpoint')
    entry = entry_for(path)
    entry['sha256'] = 'a' * 64
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('must verify hash before loading'))
    with pytest.raises(ValueError, match='SHA mismatch'):
        package.compact_checkpoint(entry, 'hard', tmp_path / 'out.pt')


def test_explicit_artifact_digests_and_duplicate_names_are_checked(tmp_path):
    path, value = full_manifest(tmp_path)
    value['artifacts'] = [{'path': str(tmp_path / 'summary.json'), 'sha256': 'a' * 64, 'name': 'summary.json'}] * 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='duplicate artifact'):
        common.load_manifest(path)


def test_archive_rejects_extra_files_and_symlinks(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    data = root / 'a.txt'
    data.write_text('data')
    common.seal_files(root)
    extra = root / 'extra.txt'
    extra.write_text('extra')
    with pytest.raises(ValueError, match='extra files'): common.verify_seal(root)
    extra.unlink()
    original = tmp_path / 'original.txt'
    data.rename(original)
    data.symlink_to(original)
    with pytest.raises(ValueError, match='SHA mismatch'): common.verify_seal(root)


@pytest.mark.parametrize('bad', ['missing', 'nonnumeric', 'standard', 'closed', 'unrelated', 'unlocked', 'shared', 'other_owner', 'replaced'])
def test_worker_lock_verifies_descriptor_inode_and_exclusive_ownership(tmp_path, monkeypatch, bad):
    lock = tmp_path / 'machine.lock'
    monkeypatch.setattr(common, 'GPU_LOCK', lock)
    monkeypatch.delenv(common.GPU_LOCK_FD_ENV, raising=False)
    if bad in ('missing', 'nonnumeric', 'standard', 'closed'):
        if bad == 'nonnumeric': monkeypatch.setenv(common.GPU_LOCK_FD_ENV, 'not-an-fd')
        if bad == 'standard': monkeypatch.setenv(common.GPU_LOCK_FD_ENV, '1')
        if bad == 'closed':
            with lock.open('a+') as f: fd = f.fileno()
            monkeypatch.setenv(common.GPU_LOCK_FD_ENV, str(fd))
        with pytest.raises(ValueError, match='GPU lock FD'):
            common.require_gpu_lock()
        return
    with lock.open('a+') as original, (tmp_path / 'unrelated.lock').open('a+') as unrelated:
        fd = unrelated.fileno() if bad == 'unrelated' else original.fileno()
        if bad in ('unrelated', 'replaced'):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if bad == 'shared': fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if bad == 'replaced':
            lock.rename(tmp_path / 'old.lock')
            lock.touch()
        with lock.open('a+') as owner:
            if bad == 'other_owner': fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            monkeypatch.setenv(common.GPU_LOCK_FD_ENV, str(fd))
            with pytest.raises(ValueError, match='GPU lock FD'):
                common.require_gpu_lock()
            if bad == 'other_owner':
                # Failed verification must never release the real owner's lock.
                with pytest.raises(RuntimeError, match='active GPU workflow'):
                    with common.gpu_lock(): pass


def test_machine_lock_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / 'target'
    target.touch()
    lock = tmp_path / 'machine.lock'
    lock.symlink_to(target)
    monkeypatch.setattr(common, 'GPU_LOCK', lock)
    with pytest.raises(OSError):
        with common.gpu_lock(): pass


def test_machine_lock_excludes_separate_checkout_and_clean_extraction(tmp_path):
    # Real production machine path, but only CPU flock calls: no runtime imports.
    # Model Hard's adopted contract with an independent raw flock implementation.
    assert common.GPU_LOCK == Path('/tmp/marso-active-gpu.lock')
    copies = []
    for name in ('other-checkout', 'clean-extraction'):
        root = tmp_path / name
        root.mkdir()
        (root / 'validation_common.py').write_bytes(Path(common.__file__).read_bytes())
        copies.append(root)
    command = [sys.executable, '-B', '-c',
               'from validation_common import gpu_lock;\nwith gpu_lock(): print("acquired")']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    with common.GPU_LOCK.open('a+') as hard:
        fcntl.flock(hard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inode = common.GPU_LOCK.stat().st_ino
        for root in copies:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and 'active GPU workflow' in result.stderr
    for root in copies:
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0 and result.stdout.strip() == 'acquired'
    with common.gpu_lock():
        result = subprocess.run([sys.executable, '-B', '-c',
            'import fcntl,sys; f=open(sys.argv[1],"a+"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
            str(common.GPU_LOCK)], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and 'BlockingIOError' in result.stderr
    assert common.GPU_LOCK.stat().st_ino == inode


def test_inherited_lock_survives_parent_close_until_worker_exits(tmp_path, monkeypatch):
    lock = tmp_path / 'machine.lock'
    monkeypatch.setattr(common, 'GPU_LOCK', lock)
    code = '''import sys
from pathlib import Path
from tools import validation_common as c
c.GPU_LOCK = Path(sys.argv[1])
c.require_gpu_lock()
print('verified', flush=True)
sys.stdin.readline()
c.require_gpu_lock()
print('still-owned', flush=True)
'''
    worker = None
    try:
        with common.gpu_lock() as fd:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{common.GPU_LOCK_FD_ENV: str(fd)})
            env.pop('PYTHONPATH', None)
            worker = subprocess.Popen([sys.executable, '-B', '-c', code, str(lock)], cwd=common.ROOT,
                                      env=env, pass_fds=(fd,), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert select.select([worker.stdout], [], [], 10)[0], 'worker did not acknowledge lock'
            assert worker.stdout.readline().strip() == 'verified'
        # The parent has closed its last descriptor; the child alone keeps it held.
        with pytest.raises(OSError): os.fstat(fd)
        with pytest.raises(RuntimeError, match='active GPU workflow'):
            with common.gpu_lock(): pass
        stdout, stderr = worker.communicate('finish\n', timeout=10)
        assert worker.returncode == 0, stderr
        assert stdout.strip() == 'still-owned'
        with common.gpu_lock() as next_fd:
            assert common.require_gpu_lock(next_fd) == next_fd
    finally:
        if worker is not None:
            if worker.poll() is None: worker.kill()
            worker.communicate(timeout=10)


def test_supervisor_passes_verified_lock_and_clears_stale_environment(tmp_path, monkeypatch):
    lock = tmp_path / 'machine.lock'
    monkeypatch.setattr(common, 'GPU_LOCK', lock)
    code = ('import sys; from pathlib import Path; from tools import validation_common as c; '
            'c.GPU_LOCK=Path(sys.argv[1]); print(c.require_gpu_lock())')
    with common.gpu_lock() as fd:
        common.supervised([sys.executable, '-B', '-c', code, str(lock)], tmp_path, 10, lock_fd=fd)
        assert (tmp_path / 'worker.log').read_text().strip() == str(fd)
        assert common.require_gpu_lock(fd) == fd
    monkeypatch.setenv(common.GPU_LOCK_FD_ENV, '12345')
    common.supervised([sys.executable, '-B', '-c',
        f'import os; assert {common.GPU_LOCK_FD_ENV!r} not in os.environ'], tmp_path, 10)
    with lock.open('a+') as unlocked:
        with pytest.raises(ValueError, match='exclusive flock'):
            common.supervised([sys.executable, '-B', '-c', 'raise AssertionError("launched")'],
                              tmp_path, 10, lock_fd=unlocked.fileno())


@pytest.mark.parametrize('script,args', [
    ('run_generalization.py', ['missing-plan.json', 'fresh_seed', 'hard']),
    ('build_repro_package.py', ['smoke', 'missing-extraction']),
])
@pytest.mark.parametrize('forged', [False, True])
def test_actual_gpu_worker_cli_rejects_missing_or_unrelated_inherited_lock(tmp_path, script, args, forged):
    out = tmp_path / 'must-not-exist'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop(common.GPU_LOCK_FD_ENV, None)
    with (tmp_path / 'repo-only.lock').open('a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if forged: env[common.GPU_LOCK_FD_ENV] = str(f.fileno())
        result = subprocess.run([sys.executable, '-B', str(common.ROOT / 'tools' / script),
                                 '--worker', *args, str(out)], env=env,
                                pass_fds=(f.fileno(),) if forged else (),
                                capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and 'GPU lock FD' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr
    assert not out.exists()


@pytest.mark.parametrize('worker', ['evaluation', 'smoke'])
def test_direct_worker_function_rejects_before_runtime_import(tmp_path, monkeypatch, worker):
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name in ('torch', 'omegaconf', 'warehouse_sort.utils'):
            pytest.fail(f'runtime import without machine lock: {name}')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.delenv(common.GPU_LOCK_FD_ENV, raising=False)
    with pytest.raises(ValueError, match='inherited GPU lock FD required'):
        if worker == 'evaluation': general.run_worker('missing', 'stress', 'hard', tmp_path)
        else: package.smoke_worker(tmp_path, tmp_path)


@pytest.mark.parametrize('worker', ['evaluation', 'smoke'])
def test_actual_worker_accepts_verified_lock_before_runtime_boundary(tmp_path, monkeypatch, worker):
    import builtins
    original_import = builtins.__import__
    class RuntimeBoundary(Exception): pass
    def guarded_import(name, *args, **kwargs):
        if name == 'torch':
            raise RuntimeBoundary('verified; stop before any GPU runtime import')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(common, 'GPU_LOCK', tmp_path / 'machine.lock')
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    with common.gpu_lock() as fd:
        monkeypatch.setenv(common.GPU_LOCK_FD_ENV, str(fd))
        with pytest.raises(RuntimeBoundary):
            if worker == 'evaluation': general.run_worker('missing', 'stress', 'hard', tmp_path)
            else: package.smoke_worker(tmp_path, tmp_path)
        assert common.require_gpu_lock(fd) == fd


@pytest.mark.parametrize('script', ['run_colab_hard.py', 'run_colab_hard_resume.py', 'run_generalization.py', 'build_repro_package.py'])
def test_preflight_explicitly_rejects_hard_and_package_supervisors(monkeypatch, script):
    def command(argv, **kwargs):
        assert argv[0] == 'ps', 'active supervisor must reject before querying GPU'
        return types.SimpleNamespace(stdout=f'424242 /content/python /other-checkout/tools/{script} --run\n')
    monkeypatch.setattr(common.subprocess, 'run', command)
    with pytest.raises(ValueError, match='processes active: \\[424242\\]'):
        common.check_idle_gpu()
