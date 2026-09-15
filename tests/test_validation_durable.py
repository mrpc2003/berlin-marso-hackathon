"""Real CPU filesystem/process evidence; no GPU runtime, real weights or installs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools import run_validation_durable as durable
from tools import run_generalization as general
from tools import validation_common as common


def synthetic_process_reader(pidfile=None):
    """Known test-owned topology, avoiding sandbox-denied macOS ps.

    This substitutes discovery only: Popen, signals, sessions, timeout, approved
    supervisor cleanup and the unrelated-process assertions execute for real.
    """
    def read(process):
        result = {process.pid: (os.getpid(), 'test-owned-parent')}
        if pidfile and pidfile.exists():
            result[int(pidfile.read_text())] = (process.pid, 'test-owned-worker')
        return result
    return read


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def layout(tmp_path, monkeypatch):
    mount = tmp_path / "mounted"
    drive = mount / "MyDrive" / "private" / "durable_new"
    drive.mkdir(parents=True)
    local = tmp_path / "local"
    records = local / "records"
    records.mkdir(parents=True)
    parent = local / "validation_runs"
    parent.mkdir()
    validation = parent / "validation_20260915-000000-123456789"
    validation.mkdir()
    monkeypatch.setattr(durable, "DRIVE_MOUNT", mount)
    monkeypatch.setattr(os.path, "ismount", lambda p: Path(p) == mount)
    dump(records / "invocation.json", {"test": True})
    return dict(records=str(records), validation_parent=str(parent), remote_dir=str(drive),
                state={"compute_status": "running", "status": "running", "remote_status": "pending"},
                source_sha256=common.source_hashes(), final=False)


def validation_of(request):
    return durable.discover_validation(Path(request["validation_parent"]))


def make_complete(request, stages=None):
    validation = validation_of(request)
    frozen = general.frozen_protocols()
    names = list(stages) if stages else list(durable.STAGES)
    inputs = {"checkpoints": {l: {"sha256": "a" * 64} for l in common.LEVELS}}
    manifest_path = Path(request["records"]) / "input_manifest.json"
    if manifest_path.exists():
        inputs = json.loads(manifest_path.read_text())
    else:
        dump(manifest_path, inputs)
    plan = dict(protocol_sha256=durable.PROTOCOL_SHA, source_sha256=request["source_sha256"],
                frozen=frozen, total_scored_episodes=100 * len(names), inputs=inputs,
                input_manifest_sha256=durable.hash_file(manifest_path))
    summary = {"status": "completed", "results": {"fresh_seed": {}, "stress": {}}}
    if stages:
        plan["stages"] = summary["stages"] = ["{}/{}".format(*s.rsplit('_', 1)) for s in names]
    dump(validation / "plan.json", plan)
    for stage in names:
        p, level = stage.rsplit('_', 1)
        spec = frozen["protocols"][p][level]
        directory = validation / stage
        directory.mkdir(exist_ok=True)
        rows = [dict(seed=s, num_parcels=spec["num_parcels"], max_episode_steps=spec["max_episode_steps"],
                     step_calls=spec["rollout_step_calls"], sorted=1, mis_sorted=0, all_placed=False,
                     steps_to_complete=spec["max_episode_steps"]) for s in spec["seeds"]]
        (directory / "episodes.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
        metrics = general.aggregate_rows(rows, spec)
        dump(directory / "result.json", dict(protocol=p, level=level, checkpoint_sha256=inputs["checkpoints"][level]["sha256"], metrics=metrics))
        summary["results"][p][level] = metrics
    dump(validation / "summary.json", summary)
    common.seal_files(validation)
    request.update(final=True, state={"compute_status": "succeeded", "status": "pending_durability", "remote_status": "pending"})
    return validation


def test_copy_independent_hash_and_destination_tamper(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "dest.json"
    source.write_bytes(os.urandom(2 * 1024 * 1024 + 91))
    digest = durable.copy_file(source, target)
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"tampered")
    assert durable.hash_file(target) != digest
    assert not list(tmp_path.glob("*.tmp"))


def test_copy_on_filesystem_requiring_closed_write_handle(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(os.urandom(1024 * 1024 + 7))
    real_open = os.open
    writers = {}
    reopened = []

    def restrictive_open(path, flags, *args, **kwargs):
        assert flags & os.O_ACCMODE != os.O_RDWR, "filesystem rejects simultaneous read/write"
        name = os.fspath(path)
        if name in writers and flags & os.O_ACCMODE == os.O_RDONLY:
            with pytest.raises(OSError):
                os.fstat(writers[name])
            reopened.append(name)
        fd = real_open(path, flags, *args, **kwargs)
        if name.endswith(".tmp") and flags & os.O_ACCMODE == os.O_WRONLY:
            writers[name] = fd
        return fd

    monkeypatch.setattr(os, "open", restrictive_open)
    assert durable.copy_file(source, target) == hashlib.sha256(source.read_bytes()).hexdigest()
    assert target.read_bytes() == source.read_bytes()
    assert len(reopened) == 1


def test_mutation_during_copy_keeps_previous_destination(tmp_path, monkeypatch):
    source, target = tmp_path / "source.json", tmp_path / "dest.json"
    source.write_bytes(b"original")
    target.write_bytes(b"previous durable bytes")
    real = durable.digest_stream
    calls = []
    def mutate(stream):
        result = real(stream)
        if not calls:
            source.write_bytes(b"modified")
        calls.append(True)
        return result
    monkeypatch.setattr(durable, "digest_stream", mutate)
    with pytest.raises(durable.TransientCopy, match="changed during copy"):
        durable.copy_file(source, target)
    assert target.read_bytes() == b"previous durable bytes"
    assert set(p.name for p in tmp_path.iterdir()) == {"source.json", "dest.json"}


@pytest.mark.parametrize("where", ["source", "source_parent", "destination", "destination_parent"])
def test_copy_rejects_symlinks_at_every_boundary(tmp_path, where):
    real = tmp_path / "real"
    real.mkdir()
    (real / "file").write_bytes(b"private")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    source, target = real / "file", tmp_path / "target"
    if where == "source":
        source = tmp_path / "source-link"
        source.symlink_to(real / "file")
    elif where == "source_parent":
        source = link / "file"
    elif where == "destination":
        target.symlink_to(real / "file")
    else:
        target = link / "newfile"
    with pytest.raises(ValueError, match="symlink"):
        durable.copy_file(source, target)
    assert (real / "file").read_bytes() == b"private"
    assert not (real / "newfile").exists()


def test_mount_path_escape_and_manifest_pending_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="escape"):
        durable.checked_path(tmp_path / "x" / ".." / "outside")
    monkeypatch.setattr(os.path, "ismount", lambda p: False)
    with pytest.raises(ValueError, match="actual.*mount"):
        durable.check_drive(Path("/content/drive/MyDrive/private"))
    monkeypatch.setattr(durable, "check_drive", lambda p: Path(p))
    # Reviewed example intentionally contains pending Hard and cannot pass even dry-run.
    with pytest.raises(ValueError):
        durable.prepare(dict(drive_parent=str(tmp_path), manifest=str(common.ROOT / "docs/VALIDATION_INPUT.example.json"), dry_run=True))


def test_preflight_checkpoint_parent_symlink_rejected(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    (real / "best.pt").write_bytes(b"synthetic")
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    manifest = tmp_path / "manifest.json"
    dump(manifest, {"checkpoints": {"easy": {"path": str(link / "best.pt")}}})
    monkeypatch.setattr(durable, "check_drive", lambda p: Path(p))
    with pytest.raises(ValueError, match="symlink"):
        durable.prepare(dict(drive_parent=str(tmp_path), manifest=str(manifest), dry_run=True))


def test_discovery_cannot_select_old_or_multiple_experiments(layout, tmp_path):
    parent = Path(layout["validation_parent"])
    old = tmp_path / "unrelated" / "validation_20260101-000000-000000000"
    old.mkdir(parents=True)
    assert validation_of(layout).parent == parent
    second = parent / "validation_20260915-000000-987654321"
    second.mkdir()
    with pytest.raises(ValueError, match="multiple"):
        durable.discover_validation(parent)
    second.rmdir()
    actual = validation_of(layout)
    actual.rmdir()
    actual.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        durable.discover_validation(parent)


def test_partial_progress_pose_records_and_transient_retry(layout, monkeypatch):
    validation = validation_of(layout)
    stage = validation / "fresh_seed_easy"
    stage.mkdir()
    row = dict(seed=6000, num_parcels=2, max_episode_steps=250, step_calls=249,
               sorted=2, mis_sorted=0, all_placed=True, steps_to_complete=50)
    (stage / "episodes.jsonl").write_text(json.dumps(row) + '\n' + '{"seed":6001')
    (stage / "diagnostics.jsonl").write_text('{"seed":6000,"diagnostic_only":true}\n')
    dump(stage / "progress.json", {"status": "running"})
    (stage / "result.json.tmp").write_text("in flight")
    (validation / ".locks").mkdir()
    (validation / ".locks" / "worker").write_text("lock")
    real_copy = durable.copy_file
    def transient(src, dst):
        if src.name == "episodes.jsonl":
            raise durable.TransientCopy("source changed during copy")
        return real_copy(src, dst)
    monkeypatch.setattr(durable, "copy_file", transient)
    first = durable.sync_cycle(layout)
    assert first["status"] == "running" and first["remote_status"] == "partial_verified"
    assert first["stages"]["fresh_seed_easy"]["episodes"] == 1
    assert first["durable_stages"]["fresh_seed_easy"]["episodes"] == 0
    assert first["transient_files"]
    remote = Path(layout["remote_dir"])
    assert (remote / validation.name / stage.name / "diagnostics.jsonl").is_file()
    assert not (remote / validation.name / ".locks").exists()
    assert not (remote / validation.name / stage.name / "result.json.tmp").exists()
    monkeypatch.setattr(durable, "copy_file", real_copy)
    second = durable.sync_cycle(layout)
    assert not second["transient_files"]
    assert second["durable_stages"]["fresh_seed_easy"]["episodes"] == 1
    assert (remote / validation.name / stage.name / "episodes.jsonl").read_bytes() == (stage / "episodes.jsonl").read_bytes()
    assert json.loads((remote / "DURABILITY_STATUS.json").read_text())["phase"] == "fresh_seed_easy"


@pytest.mark.parametrize("extra", ["checkpoints/best.pt", "auth/token.json", "demos/episode.json", "unexpected.json"])
def test_non_evidence_outputs_never_copied(layout, extra):
    path = validation_of(layout) / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"private")
    state = durable.sync_cycle(layout)
    assert state["remote_status"] == "copy_failed"
    assert not (Path(layout["remote_dir"]) / validation_of(layout).name / extra).exists()


def test_final_exact_600_and_seal_content_readback(layout):
    validation = make_complete(layout)
    state = durable.sync_cycle(layout)
    assert state["status"] == "completed" and state["remote_status"] == "final_verified"
    assert sum(v["episodes"] for v in state["stages"].values()) == 600
    assert all(v["metrics"]["n_episodes"] == 100 for v in state["stages"].values())
    assert state["seal_sha256"] == hashlib.sha256((validation / durable.SEAL).read_bytes()).hexdigest()
    remote = Path(layout["remote_dir"]) / validation.name
    assert durable.verify_payload(remote) == durable.verify_payload(validation)
    assert durable.verify_completed(validation, layout["source_sha256"], Path(layout["records"]) / "input_manifest.json") == durable.verify_payload(remote)
    # Tamper with copied payload after a valid cycle: a receipt is a point-in-time proof.
    (remote / "fresh_seed_easy" / "episodes.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        durable.verify_payload(remote)


def test_stage_subset_reports_and_completes_only_its_plan(layout):
    validation = make_complete(layout, stages=["stress_hard"])
    state = durable.sync_cycle(layout)
    assert state["status"] == "completed" and state["remote_status"] == "final_verified"
    assert list(state["stages"]) == list(state["durable_stages"]) == ["stress_hard"]
    assert state["stages"]["stress_hard"]["episodes"] == 100 and state["stages"]["stress_hard"]["metrics"]["n_episodes"] == 100
    manifest = Path(layout["records"]) / "input_manifest.json"
    receipt = durable.verify_completed(validation, layout["source_sha256"], manifest)
    assert receipt == durable.verify_payload(Path(layout["remote_dir"]) / validation.name)
    # A sealed tree whose plan claims fewer stages than it contains is not a completed subset.
    plan = json.loads((validation / "plan.json").read_text())
    (validation / "fresh_seed_easy").mkdir()
    (validation / "fresh_seed_easy" / "episodes.jsonl").write_text("")
    common.seal_files(validation)
    with pytest.raises(ValueError, match="stage directories"):
        durable.verify_completed(validation, layout["source_sha256"], manifest)
    plan["stages"] = ["stress/nope"]
    dump(validation / "plan.json", plan)
    common.seal_files(validation)
    with pytest.raises(ValueError, match="unapproved plan stages"):
        durable.verify_completed(validation, layout["source_sha256"], manifest)
    assert list(durable.progress(validation)["stages"]) == list(durable.STAGES)


def test_invalid_stage_subset_fails_before_any_directory(tmp_path):
    from types import SimpleNamespace
    args = SimpleNamespace(manifest=tmp_path / 'manifest.json', out=tmp_path / 'out', drive=tmp_path / 'drive',
                           run=True, timeout=5, sync_interval=1, sync_timeout=1, stages='stress/nope')
    with pytest.raises(ValueError, match='--stages'):
        durable.run(args)
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'drive').exists()


@pytest.mark.parametrize("failure", ["missing_seal", "bad_seal", "partial", "source_change", "input_change", "remote_extra", "tamper_on_copy", "seal_reformatted"])
def test_no_false_completed_on_final_failure(layout, monkeypatch, failure):
    validation = make_complete(layout)
    remote = Path(layout["remote_dir"]) / validation.name
    if failure == "missing_seal":
        (validation / durable.SEAL).unlink()
    elif failure == "bad_seal":
        seal = json.loads((validation / durable.SEAL).read_text())
        seal["summary.json"] = "0" * 64
        dump(validation / durable.SEAL, seal)
    elif failure == "partial":
        (validation / "fresh_seed_easy" / "episodes.jsonl").write_text("")
        common.seal_files(validation)
    elif failure == "source_change":
        monkeypatch.setattr(common, "source_hashes", lambda: {"changed.py": "a" * 64})
    elif failure == "input_change":
        dump(Path(layout["records"]) / "input_manifest.json", {'changed': True})
    elif failure == "remote_extra":
        dump(remote / "failure.json", {"unexpected": True})
    else:
        original = durable.copy_file
        def tamper(src, dst):
            result = original(src, dst)
            if failure == "tamper_on_copy" and src.name == "summary.json":
                dst.write_bytes(b"tampered")
            if failure == "seal_reformatted" and src.name == durable.SEAL:
                dst.write_text(json.dumps(json.loads(dst.read_text()), separators=(',', ':')))
            return result
        monkeypatch.setattr(durable, "copy_file", tamper)
    state = durable.sync_cycle(layout)
    assert state["status"] == "pending_durability"
    assert state["compute_status"] == "succeeded"
    assert state["remote_status"] == "copy_failed"
    assert state["copy_errors"]
    assert json.loads((Path(layout["remote_dir"]) / "DURABILITY_STATUS.json").read_text())["status"] != "completed"


def test_failure_preserves_partial_with_no_final_claim(layout):
    dump(validation_of(layout) / "failure.json", {"status": "failed", "error": "synthetic"})
    layout.update(final=True, state={"compute_status": "timed_out", "status": "failed"})
    state = durable.sync_cycle(layout)
    assert state["status"] == "failed" and state["remote_status"] == "partial_verified"
    assert "seal_sha256" not in state
    assert (Path(layout["remote_dir"]) / validation_of(layout).name / "failure.json").is_file()


def test_hanging_io_is_bounded_and_compute_continues(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    records = tmp_path / "records"
    records.mkdir()
    job = durable.IOJob(control, {}, .15, command=[sys.executable, '-c', 'import time; time.sleep(60)'])
    receipts = []
    def tick(state, final):
        if not receipts:
            result = job.poll()
            if result:
                receipts.append(result)
    start = time.monotonic()
    state = durable.supervise([sys.executable, '-u', '-c',
        'import time; print("first", flush=True); time.sleep(.6); print("after copy timeout", flush=True)'],
        records, 3, tick, process_reader=synthetic_process_reader())
    assert time.monotonic() - start < 4
    assert state["compute_status"] == "succeeded" and state["status"] == "pending_durability"
    assert receipts and not receipts[0]["ok"] and "timed out" in receipts[0]["error"]
    assert job.process.poll() is not None
    assert "after copy timeout" in (records / "child.log").read_text()


def test_io_subprocess_failure_and_real_preflight_are_bounded(tmp_path):
    job = durable.IOJob(tmp_path, {}, 2, command=[sys.executable, '-c', 'raise SystemExit(7)'])
    result = durable.bounded_job(job)
    assert not result["ok"] and "exit 7" in result["error"]
    job = durable.IOJob(tmp_path, dict(operation="prepare", manifest="missing", drive_parent="/content/drive/MyDrive/test", dry_run=True), 3)
    result = durable.bounded_job(job)
    assert not result["ok"] and "mount required" in result["error"]


def alive(pid):
    try:
        os.kill(pid, 0)
        # A dead Linux orphan can briefly await init reaping.
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.exists() and stat_path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
        return True
    except ProcessLookupError:
        return False


def test_timeout_graceful_approved_supervisor_cleans_separate_worker(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    pidfile = tmp_path / "worker.pid"
    worker = f'import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)'
    command = [sys.executable, '-c',
        f'from tools.validation_common import supervised; from pathlib import Path; import sys; supervised([sys.executable,"-c",{worker!r}],Path({str(stage)!r}),60)']
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        state = durable.supervise(command, records, 1, lambda *_: None, grace=3,
                                  process_reader=synthetic_process_reader(pidfile))
        assert state["compute_status"] == "timed_out"
        assert state["status"] == "failed"
        assert pidfile.exists() and not alive(int(pidfile.read_text()))
        assert json.loads((stage / "progress.json").read_text())["status"] == "failed"
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_owned_escalation_reaps_stubborn_setsid_descendant(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    pidfile = tmp_path / "descendant.pid"
    worker = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
    parent = ('import subprocess,sys,signal,time; from pathlib import Path; '
              'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
              f'p=subprocess.Popen([sys.executable,"-c",{worker!r}],start_new_session=True); '
              f'Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)')
    start = time.monotonic()
    state = durable.supervise([sys.executable, '-c', parent], records, 1, lambda *_: None, grace=.3,
                              process_reader=synthetic_process_reader(pidfile))
    assert time.monotonic() - start < 4
    assert state["compute_status"] == "timed_out" and state["child_returncode"] == -signal.SIGKILL
    assert not alive(int(pidfile.read_text()))


@pytest.mark.parametrize("event", ["sigterm", "pipe"])
def test_external_signal_or_closed_notebook_pipe_stops_owned_compute(tmp_path, event):
    records = tmp_path / "records"
    records.mkdir()
    pidfile = tmp_path / "child.pid"
    child = f'import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)'
    sent = []
    def tick(*_):
        if event == "sigterm" and pidfile.exists() and not sent:
            sent.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
    state = durable.supervise([sys.executable, '-c', child], records, 4, tick, grace=.5,
                             emit=(lambda _: not pidfile.exists()) if event == "pipe" else None,
                             process_reader=synthetic_process_reader())
    assert state["compute_status"] == "interrupted" and state["status"] == "failed"
    assert not alive(int(pidfile.read_text()))


def test_cli_rejects_timeout_and_arbitrary_command_and_never_changes_sources(tmp_path):
    before = common.source_hashes()
    command = [sys.executable, '-B', str(Path(durable.__file__)), '--manifest', str(tmp_path / 'manifest.json'),
               '--out', str(tmp_path / 'out'), '--drive', '/content/drive/MyDrive/test']
    for extra in [['--timeout', '43201'], ['--timeout', '0'], ['--command', 'true']]:
        result = subprocess.run(command + extra, text=True, capture_output=True, timeout=4)
        assert result.returncode != 0
    result = subprocess.run(command, text=True, capture_output=True, timeout=5)
    assert result.returncode != 0 and not (tmp_path / 'out').exists()
    assert common.source_hashes() == before
    assert hash_file_protocol() == durable.PROTOCOL_SHA
    assert 'tools/run_validation_durable.py' not in before


def hash_file_protocol():
    return hashlib.sha256((common.ROOT / 'docs/VALIDATION_PROTOCOLS.json').read_bytes()).hexdigest()


def test_wrapper_filename_is_not_rejected_as_parent(monkeypatch):
    def run(command, **kwargs):
        if command[0] == 'ps':
            return subprocess.CompletedProcess(command, 0, f'12345 {sys.executable} {durable.__file__} --run\n', '')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(common.subprocess, 'run', run)
    assert common.check_idle_gpu()['other_rollout_pids'] == []


def test_ancestry_failure_still_stops_direct_child(tmp_path):
    records = tmp_path / 'records'
    records.mkdir()
    children = []
    def unavailable(process):
        children.append(process)
        raise PermissionError('synthetic process inspection denial')
    state = durable.supervise([sys.executable, '-c', 'import time; time.sleep(60)'], records, 5,
                              lambda *_: None, grace=.2, process_reader=unavailable)
    assert state['status'] == 'failed' and state['compute_status'] == 'cleanup_pending'
    assert children and children[0].poll() is not None


def test_pid_reuse_is_not_signalled(monkeypatch):
    # A disappeared owned PID must not authorize killing its later replacement.
    sent = []
    class Process:
        pid = 101
        def poll(self): return None
    table = {101: (1, 'root-birth'), 202: (101, 'worker-birth')}
    tree = durable.OwnedTree(Process(), lambda _: table)
    tree.refresh()
    table[202] = (999, 'unrelated-new-birth')
    monkeypatch.setattr(os, 'kill', lambda pid, sig: sent.append(pid))
    tree.signal_owned(signal.SIGKILL)
    assert sent == [101]


@pytest.mark.parametrize('copy_mode,expected', [('ok', 0), ('timeout', 2), ('spawn_failure', 2)])
def test_whole_workflow_real_cpu_child_and_copy_jobs(tmp_path, monkeypatch, copy_mode, expected):
    """Full parent lifecycle with real subprocess I/O and a synthetic 600-row child.

    Only checkpoint winner pins, mount detection, and GPU computation are test
    substitutes. Production argv is captured before replacing the CPU child.
    """
    from types import SimpleNamespace
    mount = tmp_path / 'mounted'
    drive = mount / 'MyDrive' / 'private'
    drive.mkdir(parents=True)
    out = tmp_path / 'local'
    manifest = tmp_path / 'manifest.json'
    entries = {}
    for level in common.LEVELS:
        checkpoint = tmp_path / f'{level}.pt'
        checkpoint.write_bytes(f'synthetic {level} checkpoint'.encode())
        entries[level] = dict(path=str(checkpoint), sha256=durable.hash_file(checkpoint),
                              selection='best_eval_sort_accuracy', provenance_label=f'{level}-test',
                              act_horizon=8, num_inference_steps=16,
                              training_completed_iterations=common.COMPLETED[level])
    dump(manifest, {'schema_version': 1, 'checkpoints': entries, 'artifacts': []})
    real_job, real_supervise = durable.IOJob, durable.supervise
    def jobs(control, request, timeout):
        operation = request['operation']
        if operation == 'sync' and copy_mode == 'spawn_failure':
            raise OSError('synthetic cannot spawn copy')
        def command(request_path, result_path):
            code = ("from pathlib import Path; import sys,os; "
                    "from tools import run_validation_durable as d; "
                    "from tools import validation_common as c; "
                    f"d.DRIVE_MOUNT=Path({str(mount)!r}); "
                    "os.path.ismount=lambda p: Path(p)==d.DRIVE_MOUNT; c.PINNED={}; "
                    "d._job_main()")
            return [sys.executable, '-B', '-c', code, str(request_path), str(result_path)]
        if operation == 'sync' and copy_mode == 'timeout':
            return real_job(control, request, .1, command=[sys.executable, '-c', 'import time; time.sleep(60)'])
        return real_job(control, request, timeout, command=command)
    captured = []
    def cpu_supervise(command, records, timeout, tick, **kwargs):
        captured.append(command)
        parent = Path(command[command.index('--out') + 1])
        request = dict(records=str(records), validation_parent=str(parent), source_sha256=common.source_hashes())
        code = (f"import runpy,time; from pathlib import Path; p=Path({str(parent)!r}); "
                "(p/'validation_20260915-000000-123456789').mkdir(); "
                f"ns=runpy.run_path({str(Path(__file__).resolve())!r}); "
                "print('synthetic compute starts',flush=True); time.sleep(.4); "
                f"ns['make_complete']({request!r}); print('synthetic compute finished',flush=True)")
        return real_supervise([sys.executable, '-u', '-c', code], records, timeout, tick,
                              process_reader=synthetic_process_reader(), **kwargs)
    monkeypatch.setattr(durable, 'IOJob', jobs)
    monkeypatch.setattr(durable, 'supervise', cpu_supervise)
    monkeypatch.setattr(durable, 'safe_emit', lambda _: True)
    args = SimpleNamespace(manifest=manifest, out=out, drive=drive, run=True,
                           timeout=6, sync_interval=.1, sync_timeout=4)
    before = common.source_hashes()
    args.run = False
    assert durable.run(args) == 0
    assert not out.exists() and not list(drive.iterdir())
    args.run = True
    rc = durable.run(args)
    assert rc == expected
    assert len(captured) == 1
    assert captured[0][2] == str(common.ROOT / 'tools/run_generalization.py')
    assert captured[0][-1] == '--run' and '--stage-timeout' not in captured[0]
    local_runs, remote_runs = list(out.iterdir()), list(drive.iterdir())
    assert len(local_runs) == len(remote_runs) == 1
    state = json.loads((local_runs[0] / 'records/status.json').read_text())
    assert state['compute_status'] == 'succeeded'
    assert Path(state['validation_dir']).is_dir()
    assert Path(state['validation_result']).is_file()
    assert state['status'] == ('completed' if expected == 0 else 'pending_durability')
    assert common.source_hashes() == before
    if expected == 0:
        receipt = json.loads((remote_runs[0] / 'DURABILITY_STATUS.json').read_text())
        assert receipt['status'] == 'completed'
        assert durable.verify_payload(Path(state['validation_dir'])) == durable.verify_payload(Path(state['remote_validation_dir']))
        # A second launch creates new dirs, without modifying the prior run.
        old_hashes = {str(p.relative_to(remote_runs[0])): durable.hash_file(p) for p in remote_runs[0].rglob('*') if p.is_file()}
        assert durable.run(args) == 0
        assert len(list(out.iterdir())) == len(list(drive.iterdir())) == 2
        assert old_hashes == {str(p.relative_to(remote_runs[0])): durable.hash_file(p) for p in remote_runs[0].rglob('*') if p.is_file()}


def test_real_closed_stdout_pipe_interrupts_supervisor(tmp_path):
    records = tmp_path / 'records'
    records.mkdir()
    pidfile, result = tmp_path / 'child.pid', tmp_path / 'result.json'
    child_code = f'import os,time; from pathlib import Path; Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)'
    code = ("import sys,os,json; from pathlib import Path; from tools import run_validation_durable as d; "
            "os.set_blocking(sys.stdout.fileno(),False); "
            f"state=d.supervise([sys.executable,'-c',{child_code!r}],Path({str(records)!r}),5,lambda *_:None,"
            "emit=d.safe_emit,grace=.3,process_reader=lambda p:{p.pid:(os.getpid(),'owned')}); "
            f"Path({str(result)!r}).write_text(json.dumps(state))")
    process = subprocess.Popen([sys.executable, '-B', '-c', code], cwd=common.ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        process.stdout.close()
        assert process.wait(timeout=5) == 0, process.stderr.read().decode()
        state = json.loads(result.read_text())
        assert state['compute_status'] == 'interrupted' and state['status'] == 'failed'
        assert state['termination_signal'] == signal.SIGPIPE
        assert not alive(int(pidfile.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        process.stderr.close()
