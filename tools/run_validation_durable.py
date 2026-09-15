#!/usr/bin/env python3
"""Bounded, local-first durability wrapper for the approved frozen600 CLI.

No GPU imports, training, recovery/reuse, or command override is provided here.
Drive operations run only in disposable subprocesses, including preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_generalization as general
from tools import validation_common as common

ROOT = common.ROOT
PROTOCOL_SHA = "e480d74f53c7aa963d6058624c0f590e0f6fbdccbe2effad061760b958878507"
DRIVE_MOUNT = Path("/content/drive")
STAGES = tuple(f"{p}_{l}" for p in ("fresh_seed", "stress") for l in common.LEVELS)
RECORD_FILES = {"invocation.json", "input_manifest.json", "status.json", "child.log"}
TOP_FILES = {"plan.json", "preflight.json", "summary.json", "failure.json", "ARTIFACT_SHA256.json"}
STAGE_FILES = {"worker.log", "progress.json", "resolved_eval_config.json", "actual_budget.json",
               "loaded_policy.json", "diagnostics.jsonl", "episodes.jsonl", "metrics.jsonl", "result.json"}
SEAL = "ARTIFACT_SHA256.json"
GRACE_SECONDS = 25  # Approved supervisor can need two bounded 10-second waits.


class TransientCopy(ValueError):
    """An actively written source must wait for a later cycle."""


def absolute_path(value):
    p = Path(value)
    common.require(".." not in p.parts, f"path escape: {p}")
    return Path(os.path.abspath(p))


def checked_path(value):
    """Reject lexical escapes and every existing symlink, including ancestors."""
    p = absolute_path(value)
    current = Path(p.anchor)
    for part in p.parts[1:]:
        current /= part
        try:
            common.require(not stat.S_ISLNK(current.lstat().st_mode), f"symlink: {current}")
        except FileNotFoundError:
            break
    return p


def directory_fd(path, *, create=False):
    """Walk using no-follow directory FDs; never traverse a substituted symlink."""
    path = checked_path(path)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_regular(path):
    p = checked_path(path)
    parent = directory_fd(p.parent)
    try:
        fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"not a regular file: {p}")
    return os.fdopen(fd, "rb")


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def fingerprint(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def read_bytes(path):
    with open_regular(path) as stream:
        before = fingerprint(os.fstat(stream.fileno()))
        data = stream.read()
        if fingerprint(os.fstat(stream.fileno())) != before or fingerprint(Path(path).lstat()) != before:
            raise TransientCopy(f"source changed: {path}")
    return data


def read_json(path):
    return json.loads(read_bytes(path))


def hash_file(path):
    with open_regular(path) as stream:
        before = fingerprint(os.fstat(stream.fileno()))
        digest = digest_stream(stream)
        if fingerprint(os.fstat(stream.fileno())) != before or fingerprint(Path(path).lstat()) != before:
            raise TransientCopy(f"source changed: {path}")
        return digest


def atomic_bytes(path, data):
    p = checked_path(path)
    parent = directory_fd(p.parent, create=True)
    temp = f".{p.name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Reject a destination symlink instead of replacing it silently.
        checked_path(p)
        os.replace(temp, p.name, src_dir_fd=parent, dst_dir_fd=parent)
    finally:
        try:
            os.unlink(temp, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def write_json(path, value):
    atomic_bytes(path, common.json_bytes(value))


def copy_file(source, destination):
    """Commit only stable bytes; independently hash temp, source and destination."""
    source, destination = checked_path(source), checked_path(destination)
    parent = directory_fd(destination.parent, create=True)
    temp = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open_regular(source) as src:
            before = fingerprint(os.fstat(src.fileno()))
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            with os.fdopen(fd, "wb") as dst:
                h = hashlib.sha256()
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)
                    h.update(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            # DriveFS need not support read/write handles. Close the writer
            # before independently reopening the same no-follow directory entry.
            read_fd = os.open(temp, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(read_fd, "rb") as written:
                common.require(stat.S_ISREG(os.fstat(written.fileno()).st_mode), "nonregular destination temp")
                common.require(digest_stream(written) == h.hexdigest(), "destination temp SHA mismatch")
            src.seek(0)
            if (digest_stream(src) != h.hexdigest() or
                    fingerprint(os.fstat(src.fileno())) != before or fingerprint(source.lstat()) != before):
                raise TransientCopy(f"source changed during copy: {source}")
            checked_path(destination)
            os.replace(temp, destination.name, src_dir_fd=parent, dst_dir_fd=parent)
            common.require(hash_file(destination) == h.hexdigest(), "destination SHA mismatch")
            return h.hexdigest()
    finally:
        try:
            os.unlink(temp, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def ephemeral(parts):
    return any(p == ".locks" or p.endswith(".tmp") for p in parts)


def inventory(root, *, records=False):
    """Inventory only one explicitly owned tree, with an output-schema allowlist."""
    root = checked_path(root)
    found = {}
    def visit(directory, prefix=()):
        fd = directory_fd(directory)
        try:
            with os.scandir(fd) as entries:
                names = sorted(e.name for e in entries)
            for name in names:
                parts = (*prefix, name)
                mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
                common.require(not stat.S_ISLNK(mode), f"symlink in owned tree: {'/'.join(parts)}")
                if ephemeral(parts):
                    continue
                if stat.S_ISDIR(mode):
                    common.require(not records and not prefix and name in STAGES, f"unapproved directory: {parts}")
                    visit(directory / name, parts)
                else:
                    common.require(stat.S_ISREG(mode), f"nonregular output: {parts}")
                    allowed = ((len(parts) == 1 and name in RECORD_FILES) if records else
                               (len(parts) == 1 and name in TOP_FILES or
                                len(parts) == 2 and parts[0] in STAGES and
                                (name in STAGE_FILES or re.fullmatch(r"reset_seed_[0-9]+\.ppm", name))))
                    common.require(allowed, f"unapproved output: {parts}")
                    found['/'.join(parts)] = directory / name
        finally:
            os.close(fd)
    visit(root)
    return found


def discover_validation(parent):
    parent = checked_path(parent)
    fd = directory_fd(parent)
    try:
        names = os.listdir(fd)
        common.require(len(names) <= 1, "multiple entries in owned validation parent")
        if not names:
            return None
        name = names[0]
        common.require(re.fullmatch(r"validation_[0-9]{8}-[0-9]{6}-[0-9]{9}", name), "unexpected validation directory")
        common.require(stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode), "validation directory is symlink/non-directory")
        return parent / name
    finally:
        os.close(fd)


def progress(validation):
    result = {"validation_dir": str(validation) if validation else None,
              "validation_result": str(validation / "summary.json") if validation and (validation / "summary.json").is_file() else None,
              "phase": "preflight", "stages": {}}
    for stage in STAGES:
        p, level = stage.rsplit('_', 1)
        spec = general.frozen_protocols()["protocols"][p][level]
        row = {"episodes": 0, "expected_episodes": 100, "metrics": None}
        if validation:
            directory = validation / stage
            if directory.exists():
                result["phase"] = stage
                try:
                    episode_file = directory / "episodes.jsonl"
                    data = read_bytes(episode_file) if episode_file.exists() else b""
                    # A worker may be between write and newline. Count only full records.
                    rows = [json.loads(line) for line in data.split(b'\n')[:-1] if line.strip()]
                    for i, episode in enumerate(rows):
                        general.validate_episode(episode, spec)
                        common.require(i < 100 and episode["seed"] == spec["seeds"][i], "partial seed order mismatch")
                    row["episodes"] = len(rows)
                    if len(rows) == 100:
                        row["metrics"] = general.aggregate_rows(rows, spec)
                    if (directory / "result.json").exists():
                        value = read_json(directory / "result.json")
                        general.validate_metrics(value["metrics"], rows, spec)
                        row["result_metrics_verified"] = True
                except (OSError, ValueError, KeyError, TypeError) as e:
                    row["read_error"] = str(e)
        result["stages"][stage] = row
    return result


def verify_payload(root):
    files = inventory(root)
    common.require(SEAL in files, "missing final artifact seal")
    seal_bytes = read_bytes(root / SEAL)
    manifest = json.loads(seal_bytes)
    common.require(isinstance(manifest, dict) and set(manifest) == set(files) - {SEAL}, "seal payload has missing/extra files")
    for name, digest in manifest.items():
        common.require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid seal digest")
        common.require(hash_file(files[name]) == digest, f"seal SHA mismatch: {name}")
    common.require(read_bytes(root / SEAL) == seal_bytes and set(inventory(root)) == set(files), "seal/payload changed during verification")
    return {"seal_sha256": hashlib.sha256(seal_bytes).hexdigest(), "verified_files": len(manifest)}


def verify_completed(validation, expected_sources, manifest_path):
    receipt = verify_payload(validation)
    plan = read_json(validation / "plan.json")
    common.require(plan["input_manifest_sha256"] == hash_file(manifest_path) and
                   plan["inputs"] == read_json(manifest_path), "validation input snapshot mismatch")
    common.require(plan["protocol_sha256"] == PROTOCOL_SHA and plan["source_sha256"] == expected_sources,
                   "validation plan source/protocol mismatch")
    common.require(common.source_hashes() == expected_sources, "reviewed sources changed during execution")
    common.require(plan["frozen"] == general.frozen_protocols() and plan["total_scored_episodes"] == 600,
                   "validation did not use frozen600")
    summary = read_json(validation / "summary.json")
    common.require(summary.get("status") == "completed" and not (validation / "failure.json").exists(), "validation did not complete")
    for stage in STAGES:
        p, level = stage.rsplit('_', 1)
        spec = plan["frozen"]["protocols"][p][level]
        rows = [json.loads(line) for line in read_bytes(validation / stage / "episodes.jsonl").splitlines()]
        result = read_json(validation / stage / "result.json")
        common.require(result["protocol"] == p and result["level"] == level and
                       result["checkpoint_sha256"] == plan["inputs"]["checkpoints"][level]["sha256"], "stage identity mismatch")
        general.validate_metrics(result["metrics"], rows, spec)
        general.validate_metrics(summary["results"][p][level], rows, spec)
    return receipt


def check_drive(parent):
    mount, parent = checked_path(DRIVE_MOUNT), checked_path(parent)
    common.require(os.path.ismount(mount), "actual /content/drive mount required")
    common.require(parent.is_relative_to(mount / "MyDrive"), "--drive must be inside private /content/drive/MyDrive")
    return parent


def prepare(request):
    """Bounded preflight job: validate actual inputs before making a remote dir."""
    drive = check_drive(request["drive_parent"])
    manifest_path = checked_path(request["manifest"])
    original = read_bytes(manifest_path)
    raw = json.loads(original)
    for entry in list(raw.get("checkpoints", {}).values()) + raw.get("artifacts", []):
        common.require(entry.get("path") is not None, "pending input path")
        path = Path(entry["path"])
        checked_path(path if path.is_absolute() else manifest_path.parent / path)
    manifest = common.load_manifest(manifest_path, allow_pending=False)
    for level, entry in manifest["checkpoints"].items():
        common.require(hash_file(entry["path"]) == entry["sha256"], f"{level}: checkpoint SHA mismatch")
    common.require(original == read_bytes(manifest_path), "manifest changed during preflight")
    common.require(hash_file(ROOT / "docs/VALIDATION_PROTOCOLS.json") == PROTOCOL_SHA, "frozen protocol SHA changed")
    sources = common.source_hashes()
    if request["dry_run"]:
        return {"status": "dry_run", "protocol_sha256": PROTOCOL_SHA, "source_sha256": sources,
                "wrapper_sha256": hash_file(Path(__file__)), "episodes": 600}
    records = Path(request["records"])
    write_json(records / "input_manifest.json", manifest)
    remote = drive / request["name"]
    fd = directory_fd(drive, create=True)
    try:
        os.mkdir(request["name"], mode=0o700, dir_fd=fd)  # Never exist_ok: no past run reuse.
    finally:
        os.close(fd)
    return {"remote_dir": str(remote), "source_sha256": sources,
            "manifest_sha256": hashlib.sha256(original).hexdigest(), "wrapper_sha256": hash_file(Path(__file__))}


def sync_cycle(request):
    """All Drive reads, writes and final hash verification stay in this process."""
    remote = checked_path(request["remote_dir"])
    common.require(remote.parent == check_drive(remote.parent), "invalid Drive parent")
    # Remote run was exclusively created during preflight; never recreate it on unmount.
    os.close(directory_fd(remote))
    records = checked_path(request["records"])
    validation = discover_validation(Path(request["validation_parent"]))
    state = dict(request["state"])
    state.update(progress(validation))
    state["remote_validation_dir"] = str(remote / validation.name) if validation else None
    state["remote_validation_result"] = str(remote / validation.name / "summary.json") if state["validation_result"] else None
    transient, errors, copied = [], [], {}
    for tree, destination, is_records in ((records, remote / "records", True),
                                           (validation, remote / validation.name if validation else None, False)):
        if tree is None:
            continue
        try:
            files = inventory(tree, records=is_records)
        except (OSError, ValueError) as e:
            errors.append(str(e))
            continue
        for name, source in files.items():
            try:
                copied[str(source.relative_to(records.parent))] = copy_file(source, destination / name)
            except TransientCopy as e:
                transient.append(str(e))
            except (OSError, ValueError) as e:
                errors.append(str(e))
    state.update(remote_status="partial_verified" if not errors else "copy_failed",
                 transient_files=transient, copy_errors=errors, copied_files=len(copied),
                 cycle_unix=time.time())
    remote_validation = remote / validation.name if validation else None
    state["durable_stages"] = progress(remote_validation if remote_validation and remote_validation.exists() else None)["stages"]
    if not errors:
        state["last_successful_cycle_unix"] = state["cycle_unix"]
    if request["final"] and state["compute_status"] == "succeeded":
        state["status"] = "pending_durability"
        try:
            common.require(validation is not None and not errors and not transient, "final copy incomplete")
            local = verify_completed(validation, request["source_sha256"], records / "input_manifest.json")
            distant = verify_payload(remote / validation.name)
            common.require(local == distant, "local/Drive seal SHA differs")
            # Recheck local after the remote reads, including seal identity.
            common.require(verify_payload(validation) == local, "local payload changed after copy")
            state.update(status="completed", remote_status="final_verified", **local)
        except (OSError, ValueError, KeyError, TypeError) as e:
            state["copy_errors"].append(str(e))
            state["remote_status"] = "copy_failed"
    # Separate receipt: never modify the approved validation tree or its seal.
    data = common.json_bytes(state)
    atomic_bytes(remote / "DURABILITY_STATUS.json", data)
    common.require(read_bytes(remote / "DURABILITY_STATUS.json") == data, "Drive status readback mismatch")
    return state


def _job_main():
    request_path, result_path = map(Path, sys.argv[1:])
    try:
        request = read_json(request_path)
        result = {"ok": True, "value": {"prepare": prepare, "sync": sync_cycle}[request["operation"]](request)}
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    write_json(result_path, result)


class IOJob:
    """Nonblocking parent-side job handle. Kill only this owned session on expiry."""
    def __init__(self, control, request, timeout, *, command=None):
        token = uuid.uuid4().hex
        self.result = control / f"{token}.result.json"
        request_path = control / f"{token}.request.json"
        write_json(request_path, request)
        self.request_path = request_path
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        env.pop(common.GPU_LOCK_FD_ENV, None)
        if callable(command):  # Lower-level test seam; never exposed by the CLI.
            command = command(request_path, self.result)
        self.process = subprocess.Popen(command or [sys.executable, "-B", "-c",
            "from tools.run_validation_durable import _job_main; _job_main()", str(request_path), str(self.result)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        self.deadline = time.monotonic() + timeout
        self.finished = False

    def cancel(self):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return False  # e.g. uninterruptible FUSE I/O: never wait indefinitely.
        return True

    def poll(self):
        if self.process.poll() is None and time.monotonic() < self.deadline:
            return None
        self.finished = True
        if self.process.poll() is None:
            reaped = self.cancel()
            return {"ok": False, "error": "copy/preflight timed out", "reaped": reaped}
        if not self.result.exists():
            return {"ok": False, "error": f"I/O subprocess exit {self.process.returncode} without receipt"}
        try:
            return read_json(self.result)
        except (OSError, ValueError) as e:
            return {"ok": False, "error": f"invalid I/O receipt: {e}"}


def process_table(process=None):
    """PID birth identity avoids killing a reused PID; never select by command."""
    if sys.platform.startswith("linux"):
        result = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(')', 1)[1].split()
                result[int(entry.name)] = (int(fields[1]), fields[19])
            except (OSError, ValueError, IndexError):
                continue
        return result
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,lstart="], check=True,
                            capture_output=True, text=True, timeout=1).stdout
    return {int(f[0]): (int(f[1]), ' '.join(f[2:])) for line in output.splitlines() if len(f := line.split()) == 7}


class OwnedTree:
    def __init__(self, process, reader=process_table):
        self.process = process
        self.reader = reader
        self.identities = {}

    def refresh(self):
        table = self.reader(self.process)
        root = self.process.pid
        if root not in self.identities and self.process.poll() is None and root in table:
            self.identities[root] = table[root][1]
        changed = True
        while changed:
            changed = False
            for pid, (parent, birth) in table.items():
                if (pid not in self.identities and parent in self.identities and
                        table.get(parent, (None, None))[1] == self.identities[parent]):
                    self.identities[pid] = birth
                    changed = True
        return table

    def signal_owned(self, sig, *, root_only=False):
        # Popen retains direct-child ownership even when ancestry inspection fails.
        if root_only:
            self.process.send_signal(sig)
            return
        table = self.refresh()
        for pid, birth in self.identities.items():
            if table.get(pid, (None, None))[1] == birth:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass


def supervise(command, records, timeout, tick, *, grace=GRACE_SECONDS, emit=None, process_reader=process_table):
    """Lower-level CPU-test seam; the CLI always constructs the approved command."""
    stop = []
    previous = {s: signal.signal(s, lambda signum, frame: stop.append(signum))
                for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    child = None
    tree = None
    term_at = None
    start = time.monotonic()
    state = {"compute_status": "running", "status": "running", "remote_status": "pending"}
    next_emit = start
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        env.pop("PYTHONPATH", None)
        env.pop(common.GPU_LOCK_FD_ENV, None)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        os.set_blocking(child.stdout.fileno(), False)
        tree = OwnedTree(child, process_reader)
        with (records / "child.log").open("xb") as log:
            while True:
                now = time.monotonic()
                tree.refresh()
                # Drain a bounded amount each turn so a noisy child cannot starve deadlines.
                for _ in range(16):
                    try:
                        chunk = os.read(child.stdout.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    log.write(chunk)
                log.flush()
                state["elapsed_seconds"] = round(now - start, 2)
                if emit and now >= next_emit:
                    if not emit(state):
                        stop.append(signal.SIGPIPE)
                    next_emit = now + 1
                if term_at is None and (stop or now - start >= timeout):
                    state.update(compute_status="interrupted" if stop else "timed_out", status="failed",
                                 termination_signal=stop[0] if stop else None)
                    # Let the approved supervisor clean its separately sessioned worker first.
                    tree.signal_owned(signal.SIGTERM, root_only=True)
                    term_at = now
                tick(state, False)
                rc = child.poll()
                if rc is not None:
                    if term_at is None:
                        state.update(compute_status="succeeded" if rc == 0 else "failed",
                                     status="pending_durability" if rc == 0 else "failed")
                    state["child_returncode"] = rc
                    break
                if term_at is not None and now - term_at >= grace:
                    tree.signal_owned(signal.SIGKILL)
                    try:
                        state["child_returncode"] = child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        state["cleanup_pending"] = True
                    break
                time.sleep(.1)
            os.fsync(log.fileno())
    except Exception as e:
        state.update(compute_status="failed", status="failed", supervisor_error=f"{type(e).__name__}: {e}")
    finally:
        if tree:
            # Also cover local disk errors / failed tick / unexpected supervisor exit.
            child.terminate()
            if child.poll() is None:
                end = (term_at if term_at is not None else time.monotonic()) + grace
                while child.poll() is None and time.monotonic() < end:
                    try:
                        tree.refresh()
                    except (OSError, subprocess.SubprocessError) as e:
                        state["ancestry_error"] = str(e)
                    time.sleep(.1)
            try:
                tree.signal_owned(signal.SIGKILL)
            except (OSError, subprocess.SubprocessError) as e:
                state.update(ancestry_error=str(e), cleanup_pending=True)
        if child:
            if child.poll() is None:
                child.kill()
            child.stdout.close()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                state["cleanup_pending"] = True
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if state.get("cleanup_pending"):
        state.update(status="failed", compute_status="cleanup_pending")
    return state


def safe_emit(value):
    try:
        os.write(sys.stdout.fileno(), (json.dumps(value, sort_keys=True) + '\n').encode())
        return True
    except BlockingIOError:
        return True  # Output backpressure must not block supervision.
    except (BrokenPipeError, OSError):
        return False


def bounded_job(job, interruptions=()):
    try:
        while (result := job.poll()) is None:
            if interruptions:
                return {"ok": False, "error": f"I/O wait interrupted by signal {interruptions[0]}"}
            time.sleep(.1)
        return result
    finally:
        if not job.finished:
            job.cancel()


def _run(args, interruptions):
    out = absolute_path(args.out)
    common.require(not out.is_relative_to(DRIVE_MOUNT) and "drive" not in [x.lower() for x in out.parts], "--out must be local, outside Drive")
    out = checked_path(out)
    name = "durable_" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex
    # Dry run leaves no persistent local/Drive directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="marso-durable-preflight-") as scratch:
        control = checked_path(Path(scratch).resolve())
        request = dict(operation="prepare", drive_parent=str(absolute_path(args.drive)), manifest=str(absolute_path(args.manifest)),
                       dry_run=not args.run, name=name)
        if args.run:
            parent_fd = directory_fd(out, create=True)
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            local = out / name
            records, validation_parent = local / "records", local / "validation_runs"
            records.mkdir(mode=0o700)
            validation_parent.mkdir(mode=0o700)
            request["records"] = str(records)
        result = bounded_job(IOJob(control, request, args.sync_timeout), interruptions)
        if not result["ok"]:
            if args.run:
                write_json(records / "status.json", {"status": "failed", "compute_status": "not_started", "remote_status": "pending", "error": result["error"]})
                safe_emit({"local_dir": str(local), "status_file": str(records / "status.json"), "compute_status": "not_started"})
            raise ValueError(result["error"])
        if not args.run:
            safe_emit(result["value"])
            return 0
        prep = result["value"]
        if interruptions:
            write_json(records / "status.json", {"status": "failed", "compute_status": "not_started",
                       "remote_status": "pending", "termination_signal": interruptions[0]})
            return 1
        command = [sys.executable, "-B", str(ROOT / "tools/run_generalization.py"),
                   "--manifest", str(records / "input_manifest.json"), "--out", str(validation_parent), "--run"]
        invocation = dict(prep, command=command, local_dir=str(local), records=str(records),
                          validation_parent=str(validation_parent), timeout=args.timeout,
                          sync_interval=args.sync_interval, sync_timeout=args.sync_timeout)
        write_json(records / "invocation.json", invocation)
        safe_emit({"local_dir": str(local), "remote_dir": prep["remote_dir"], "status_file": str(records / "status.json")})
        active = None
        next_sync = 0
        remote_state = {"remote_status": "pending"}
        stalled = False

        def tick(state, final):
            nonlocal active, next_sync, remote_state, stalled
            if not active and not final and time.monotonic() < next_sync:
                return
            if active:
                receipt = active.poll()
                if receipt is None:
                    return
                if receipt["ok"]:
                    remote_state = receipt["value"]
                else:
                    remote_state = {"remote_status": "copy_failed", "copy_errors": [receipt["error"]]}
                    stalled = not receipt.get("reaped", True)
                active = None
            current = dict(state)
            current.update(remote_status=remote_state["remote_status"], copy_errors=remote_state.get("copy_errors", []))
            current.update(local_dir=str(local), remote_dir=prep["remote_dir"],
                           local_status_file=str(records / "status.json"), remote_status_file=str(Path(prep["remote_dir"]) / "DURABILITY_STATUS.json"))
            if remote_state.get("last_successful_cycle_unix"):
                current["last_successful_cycle_unix"] = remote_state["last_successful_cycle_unix"]
            current.update(progress(discover_validation(validation_parent)))
            write_json(records / "status.json", current)
            state.update(current)
            if not stalled and (final or time.monotonic() >= next_sync):
                next_sync = time.monotonic() + args.sync_interval
                try:
                    active = IOJob(control, dict(operation="sync", records=str(records), validation_parent=str(validation_parent),
                                   remote_dir=prep["remote_dir"], source_sha256=prep["source_sha256"],
                                   state=current, final=final), args.sync_timeout)
                except (OSError, subprocess.SubprocessError) as e:
                    remote_state = {"remote_status": "copy_failed", "copy_errors": [f"cannot start copy: {e}"]}
                    state.update(remote_state)

        try:
            state = supervise(command, records, args.timeout, tick, emit=safe_emit)
            # A running partial cycle is cancelled before the single bounded final attempt.
            if active:
                stalled = not active.cancel()
                active = None
            tick(state, True)
            if active:
                receipt = bounded_job(active, interruptions)
                active = None
                if receipt["ok"]:
                    state = receipt["value"]
                else:
                    state.update(remote_status="copy_failed", copy_errors=[receipt["error"]])
            else:
                state.update(remote_status="copy_failed", copy_errors=(["prior I/O child unreaped; no overlapping writers"]
                             if stalled else remote_state.get("copy_errors", ["final copy did not start"])))
            if state["compute_status"] == "succeeded" and state.get("remote_status") != "final_verified":
                state["status"] = "pending_durability"
            state.update(local_dir=str(local), remote_dir=prep["remote_dir"], **progress(discover_validation(validation_parent)))
            write_json(records / "status.json", state)
            safe_emit(state)
            return 0 if state["status"] == "completed" else (2 if state["compute_status"] == "succeeded" else 1)
        finally:
            if active:
                active.cancel()


def run(args):
    interruptions = []
    previous = {s: signal.signal(s, lambda signum, frame: interruptions.append(signum))
                for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        return _run(args, interruptions)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--drive", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--sync-interval", type=int, default=30)
    parser.add_argument("--sync-timeout", type=int, default=60)
    args = parser.parse_args(argv)
    for name in ("timeout", "sync_interval", "sync_timeout"):
        common.require(0 < getattr(args, name) <= 43200, f"{name} must be 1..43200 seconds")
    os.set_blocking(sys.stdout.fileno(), False)
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        safe_emit({"status": "failed", "error": str(error)})
        sys.exit(1)
