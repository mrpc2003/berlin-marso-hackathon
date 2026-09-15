"""Real CPU process/FD/signal tests; all signalled process groups are created by these fixtures."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from tools import run_colab_hard as hard

REPO = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('CPU subprocess fixture did not reach expected state')
        time.sleep(0.02)


def pid_live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform.startswith('linux'):
        try:
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    return True


@pytest.mark.parametrize('mode', ['leader_exit', 'timeout', 'supervisor_signal'])
def test_real_descendants_cleaned_after_leader_exit_timeout_and_signal(tmp_path, mode):
    root = tmp_path.resolve()
    grandchild = "import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path('grandchild.pid').write_text(str(os.getpid())); time.sleep(60)"
    child = ("import os, pathlib, signal, subprocess, sys, time\n"
             "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
             "pathlib.Path('child.pid').write_text(str(os.getpid()))\n"
             f"p=subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
             "time.sleep(60)\n")
    dispatcher = ("import pathlib, subprocess, sys, time\n"
                  f"p=subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                  "while not pathlib.Path('grandchild.pid').exists(): time.sleep(.01)\n" +
                  ("sys.exit(0)\n" if mode == 'leader_exit' else "time.sleep(60)\n"))
    supervisor = ("import json, os, pathlib, signal, sys\nfrom tools import run_colab_hard as hard\n"
                  # This sandbox disallows ps. Bypass only the independent duplicate-GPU scan;
                  # group liveness, process creation, FD inheritance and all signals remain real.
                  "hard.reject_gpu_processes=lambda: None\n"
                  "def interrupted(*args): raise KeyboardInterrupt('fixture SIGTERM')\n"
                  "signal.signal(signal.SIGTERM, interrupted)\n"
                  "original_stop=hard.stop_owned_process\n"
                  "hard.stop_owned_process=lambda p: original_stop(p, grace=.2, kill_grace=3)\n"
                  "def update(**kw):\n"
                  " if kw.get('stage_pid'): pathlib.Path('group.pid').write_text(str(kw['stage_pid']))\n"
                  f"hard.run_stage([sys.executable, '-c', {dispatcher!r}], pathlib.Path.cwd()/'stage.log', "
                  f"{1 if mode == 'timeout' else 30}, update, os.environ.copy(), str(pathlib.Path.cwd()))\n")
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1')
    with (root / 'supervisor.log').open('w') as log:
        p = subprocess.Popen([sys.executable, '-c', supervisor], cwd=root, env=env, stdout=log, stderr=log,
                             start_new_session=True)
    try:
        wait_for(lambda: (root / 'grandchild.pid').exists())
        if mode == 'supervisor_signal':
            p.send_signal(signal.SIGTERM)
        p.wait(timeout=15)
        assert p.returncode != 0  # a leftover descendant cannot certify stage success
        for name in ('child.pid', 'grandchild.pid'):
            pid = int((root / name).read_text())
            wait_for(lambda: not pid_live(pid))
        assert unrelated.poll() is None
        assert not hard.owned_group_live(int((root / 'group.pid').read_text()))
    finally:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=5)
        if (root / 'group.pid').exists():
            try:
                os.killpg(int((root / 'group.pid').read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_unlocked_and_wrong_open_description_are_rejected(tmp_path, monkeypatch):
    repo, local = tmp_path.resolve() / 'repo', tmp_path.resolve() / 'local'
    repo.mkdir(); local.mkdir()
    monkeypatch.setattr(hard, 'reject_gpu_processes', lambda: None)
    with hard.launch_locks(repo, local) as fds:
        hard.verify_inherited_locks(fds, repo, local)
        with hard.lock_paths(repo, local)[0].open('a') as wrong:
            with pytest.raises(ValueError, match='does not own'):
                hard.verify_inherited_locks((wrong.fileno(), *fds[1:]), repo, local)
        fcntl.flock(fds[-1], fcntl.LOCK_UN)
        with pytest.raises(ValueError, match='not already held'):
            hard.verify_inherited_locks(fds, repo, local)


def test_actual_worker_retains_machine_lock_after_parent_handles_close(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    repo, local = root / 'repo', root / 'local'
    repo.mkdir(); local.mkdir()
    (repo / 'eval.py').write_text(
        "import os, pathlib, sys, time\n"
        "assert 'torch' not in sys.modules, 'FD verification must precede CUDA-capable imports'\n"
        "pathlib.Path('worker.pid').write_text(str(os.getpid()))\n"
        "while not pathlib.Path('finish').exists(): time.sleep(.02)\n")
    monkeypatch.setattr(hard, 'reject_gpu_processes', lambda: None)
    with hard.launch_locks(repo, local) as fds:
        p = subprocess.Popen([sys.executable, str(REPO / 'tools/hard_gpu_worker.py'), '--repo', str(repo),
                              '--local', str(local), '--lock-fds', *map(str, fds), 'eval.py'], cwd=repo,
                             env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1'),
                             pass_fds=fds, start_new_session=True)
        wait_for(lambda: (repo / 'worker.pid').exists() or p.poll() is not None)
        assert p.poll() is None
    try:
        assert int((repo / 'worker.pid').read_text()) == p.pid
        other_repo, other_local = root / 'other-repo', root / 'other-local'
        other_repo.mkdir(); other_local.mkdir()
        with pytest.raises(RuntimeError, match='marso-active-gpu.lock'):
            with hard.launch_locks(other_repo, other_local):
                pytest.fail('Machine-wide lock must block a different checkout')
        for path in hard.lock_paths(repo, local):
            with path.open('a') as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (repo / 'finish').touch()
        assert p.wait(timeout=5) == 0
        with hard.launch_locks(other_repo, other_local):
            pass
    finally:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=5)


def test_real_hydra_wrapper_preserves_flags_config_logs_and_trainer_pid(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    repo, local = root / 'repo', root / 'local'
    repo.mkdir(); local.mkdir()
    shutil.copytree(REPO / 'il/conf', repo / 'il/conf')
    shutil.copy2(REPO / 'il/train.py', repo / 'il/train.py')
    trainer = repo / 'il/baselines/diffusion_policy/train_rgbd.py'
    trainer.parent.mkdir(parents=True)
    trainer.write_text("import json, os, pathlib, sys\n"
                       "from tools import run_colab_hard as hard\n"
                       "fds=json.loads(os.environ['FIXTURE_FDS'])\n"
                       "hard.verify_inherited_locks(fds, os.environ['FIXTURE_REPO'], os.environ['MARSO_HARD_LOCAL_ROOT'])\n"
                       "pathlib.Path(os.environ['MARSO_HARD_LOCAL_ROOT'],'actual.json').write_text(json.dumps(dict(pid=os.getpid(), args=sys.argv[1:], cwd=os.getcwd())))\n")
    demo = repo / hard.DEMO
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b'CPU fixture sentinel; not real data')
    (repo / 'tools').mkdir()
    shutil.copy2(REPO / 'tools/hard_gpu_worker.py', repo / 'tools/hard_gpu_worker.py')
    monkeypatch.setattr(hard, 'reject_gpu_processes', lambda: None)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo), str(REPO))), PYTHONDONTWRITEBYTECODE='1',
               MARSO_HARD_LOCAL_ROOT=str(local), FIXTURE_REPO=str(repo))
    plan = dict(python=sys.executable, repo=str(repo), local_root=str(local))
    events = []
    with hard.launch_locks(repo, local) as fds:
        env['FIXTURE_FDS'] = json.dumps(fds)
        hard.run_stage(hard.train_command(plan), local / 'stage.log', 20, lambda **kw: events.append(kw),
                       env, str(repo), lock_fds=fds)
    actual = json.loads((local / 'actual.json').read_text())
    launch = json.loads((local / 'train/hydra/launcher.json').read_text())
    assert actual['pid'] == launch['pid'] == next(e['stage_pid'] for e in events if e.get('stage_pid'))
    assert actual['args'] == launch['command'][3:]
    assert actual['cwd'] == str(trainer.parent)
    from omegaconf import OmegaConf
    from il.train import _flags_to_cli
    cfg = OmegaConf.load(local / 'train/hydra/resolved-config.yaml')
    assert cfg.flags.total_iters == 40000 and cfg.flags.resume is None and cfg.flags.seed == 1
    assert cfg.flags.amp and cfg.flags.batch_size == 128 and cfg.max_episode_steps == 800
    expected = ['--demo-path', str(demo), '--env-id', cfg.env_id, '--control-mode', cfg.control_mode,
                '--sim-backend', cfg.sim_backend, '--max-episode-steps', '800']
    assert actual['args'] == expected + _flags_to_cli(OmegaConf.to_container(cfg.flags, resolve=True))
    assert launch['hydra_overrides'] == hard.train_command(plan)[3:-1]
    assert '[il/train] method=dp_rgb_hard' in (local / 'stage.log').read_text()
    assert (local / 'train/hydra/hard_gpu_worker.log').exists()
