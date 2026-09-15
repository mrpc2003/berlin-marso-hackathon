"""Small local contracts shared by validation and release commands (no runtime imports)."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("easy", "medium", "hard")
BUDGETS = dict(zip(LEVELS, (250, 500, 800)))
PARCELS = dict(zip(LEVELS, (2, 4, 6)))
COMPLETED = dict(zip(LEVELS, (25000, 30000, 40000)))
WEIGHTS = dict(zip(LEVELS, (.2, .3, .5)))
PINNED = {
    "easy": ("c563ab461b4b1ad87f5a3d2d77df8453334eb6d8a3e6a86f20b7b42676352df9", 8, 32),
    "medium": ("95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb", 8, 16),
}
ENTRYPOINT = "warehouse_sort.il_policy:load_dp_rgb"
GPU_LOCK = Path("/tmp/marso-active-gpu.lock")
GPU_LOCK_FD_ENV = "MARSO_GPU_LOCK_FD"
DEPENDENCIES = ("torch", "torchvision", "mani-skill", "sapien", "diffusers", "gymnasium",
                "hydra-core", "omegaconf", "numpy", "transforms3d", "tyro", "matplotlib",
                "tensorboard", "h5py", "tqdm")
SOURCE_FILES = (
    "eval.py", "pyproject.toml", "pixi.toml", "pixi.lock", "SUBMISSION.md",
    "tools/strip_ckpt.py", "tools/rollout_logger.py", "tools/validation_common.py",
    "tools/run_generalization.py", "tools/build_repro_package.py",
    "docs/VALIDATION_PROTOCOL.md", "docs/VALIDATION_PROTOCOLS.json", "docs/REPRODUCE.md",
    "docs/VALIDATION_INPUT.example.json",
)
SOURCE_TREES = {"warehouse_sort": ".py", "conf": ".yaml", "il/conf": ".yaml",
                "il/baselines/diffusion_policy/diffusion_policy": ".py"}
EXTRA_SOURCE = ("il/baselines/diffusion_policy/train_rgbd.py", "il/baselines/diffusion_policy/setup.py",
                "il/baselines/diffusion_policy/README.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def new_output(parent, prefix):
    parent = Path(parent).resolve()
    require("drive" not in [p.lower() for p in parent.parts], "--out must be local; parent copies to Drive later")
    parent.mkdir(parents=True, exist_ok=True)
    name = prefix + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{time.time_ns() % 1000000000:09d}"
    out = parent / name
    out.mkdir(exist_ok=False)
    return out


def safe_relative(name):
    p = Path(name)
    forbidden = {"auth", "demos", "cache", "caches", "outputs", "runs", "__pycache__"}
    require(not p.is_absolute() and bool(p.parts), "expected a relative allowlisted path")
    require(all(x not in (".", "..") and not x.startswith(".") and x.lower() not in forbidden
                for x in p.parts), f"excluded path: {name}")
    return p


def source_files(root=ROOT):
    root = Path(root)
    names = set(SOURCE_FILES + EXTRA_SOURCE)
    for tree, suffix in SOURCE_TREES.items():
        names.update(str(p.relative_to(root)) for p in (root / tree).rglob("*" )
                     if p.is_file() and p.suffix == suffix and "__pycache__" not in p.parts)
    # Preserve existing license texts if present; do not invent a project license.
    for directory in (root, *(root / tree for tree in SOURCE_TREES)):
        names.update(str(p.relative_to(root)) for p in directory.glob("LICENSE*") if p.is_file())
    for name in sorted(names):
        rel = safe_relative(name)
        path = root / rel
        require(path.is_file() and not path.is_symlink(), f"missing or symlinked source: {name}")
        require(path.resolve().is_relative_to(root.resolve()), f"source escapes root: {name}")
    return sorted(names)


def source_hashes(root=ROOT):
    return {name: sha256(Path(root) / name) for name in source_files(root)}


def load_manifest(path, *, allow_pending=False, verify_files=False):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    require(value.get("schema_version") == 1, "manifest schema_version must be 1")
    require(set(value.get("checkpoints", {})) == set(LEVELS), "explicit checkpoint mapping must contain all three levels")
    for level, entry in value["checkpoints"].items():
        require(entry.get("selection") == "best_eval_sort_accuracy", f"{level}: must select verified best checkpoint")
        label = entry.get("provenance_label", "")
        require(isinstance(label, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", label), "use a stable provenance label, not a path")
        fields = ("path", "sha256", "act_horizon", "num_inference_steps", "training_completed_iterations")
        if allow_pending and level == "hard" and any(entry.get(k) is None for k in fields):
            continue
        require(all(entry.get(k) is not None for k in fields), f"{level}: pending checkpoint or winner settings")
        require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), f"{level}: invalid sha256")
        require(entry["training_completed_iterations"] == COMPLETED[level], f"{level}: completed training budget must be {COMPLETED[level]}")
        for key in ("act_horizon", "num_inference_steps"):
            require(type(entry[key]) is int and 1 <= entry[key] <= 100, f"{level}: invalid {key}")
        if level in PINNED:
            require((entry["sha256"], entry["act_horizon"], entry["num_inference_steps"]) == PINNED[level], f"{level}: does not match reviewed winner")
        source = Path(entry["path"])
        if not source.is_absolute():
            source = path.parent / source
        entry["path"] = str(source.resolve())
        if verify_files:
            require(source.is_file() and not source.is_symlink(), f"{level}: checkpoint missing or symlinked")
            require(sha256(source) == entry["sha256"], f"{level}: checkpoint SHA mismatch")
    for artifact in value.get("artifacts", []):
        name = safe_relative(artifact["name"])
        require(name.suffix in (".json", ".jsonl", ".png", ".ppm", ".md", ".yaml", ".txt"), "unsupported explicit artifact type")
        p = Path(artifact["path"])
        if not p.is_absolute():
            p = path.parent / p
        artifact["path"] = str(p.resolve())
        if verify_files:
            require(p.is_file() and not p.is_symlink() and sha256(p) == artifact["sha256"], f"artifact hash mismatch: {name}")
    names = [a["name"] for a in value.get("artifacts", [])]
    require(len(names) == len(set(names)), "duplicate artifact names")
    return value


def checkpoint_contract(ckpt, level, entry):
    """Use stored config demo lineage and budget, never legacy args.num_parcels."""
    import torch
    cfg = ckpt.get("config", {})
    require(cfg.get("obs_mode") == "rgb" and cfg.get("act_dim") == 4 and cfg.get("state_dim") == 26,
            f"{level}: expected RGB + 26 proprio features and Nx4 actions")
    require(cfg.get("max_episode_steps") == BUDGETS[level], f"{level}: max_episode_steps mismatch")
    demos = cfg.get("demo_paths", [])
    require(isinstance(demos, list) and demos, f"{level}: missing demo_paths lineage")
    for demo in demos:
        parts = Path(str(demo)).parts
        seen = set(parts) & set(LEVELS)
        require(seen == {level}, f"{level}: wrong or ambiguous demo path level")
    require(type(ckpt.get("iteration")) is int and 0 <= ckpt["iteration"] <= COMPLETED[level], f"{level}: invalid best iteration")
    require(cfg.get("total_iters") is not None and 0 < cfg["total_iters"] <= COMPLETED[level], f"{level}: excessive or missing training budget")
    ema = ckpt.get("ema_agent") or ckpt.get("model")
    require(isinstance(ema, dict) and bool(ema), f"{level}: verified best EMA (ema_agent/model) required; raw agent fallback forbidden")
    for key, tensor in ema.items():
        require(isinstance(tensor, torch.Tensor), f"non-tensor EMA value: {key}")
        require(not tensor.is_complex() and bool(torch.isfinite(tensor).all()), f"nonfinite/complex EMA: {key}")
        require(not tensor.is_floating_point() or tensor.dtype == torch.float32, f"{key}: source EMA must already be float32 for exact identity")
    require(entry["act_horizon"] <= cfg["pred_horizon"] - cfg["obs_horizon"] + 1, "act_horizon exceeds available prediction chunk")
    require(entry["num_inference_steps"] <= cfg["num_diffusion_iters"], "inference steps exceed training diffusion schedule")
    return cfg


def versions():
    rows = {}
    for name in DEPENDENCIES:
        try:
            dist = importlib.metadata.distribution(name)
            rows[name] = {"version": dist.version, "license": dist.metadata.get("License-Expression") or dist.metadata.get("License") or "not declared"}
        except importlib.metadata.PackageNotFoundError:
            rows[name] = {"version": None, "license": "not installed"}
    return {"python": sys.version, "packages": rows}


def _lock_file(path):
    """Open the permanent regular inode; never follow a replaced symlink."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "GPU lock must be a regular file")
        return os.fdopen(fd, "a+")
    except BaseException:
        os.close(fd)
        raise


def _same_lock_inode(fd, path):
    actual, expected = os.fstat(fd), Path(path).lstat()
    require(stat.S_ISREG(actual.st_mode) and stat.S_ISREG(expected.st_mode) and
            (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
            "GPU lock FD does not refer to the permanent machine lock inode")


def require_gpu_lock(lock_fd=None):
    """Verify an inherited exclusive flock before any GPU runtime import.

    An FD number, matching inode, or someone else's lock alone is insufficient.
    An independent shared-lock probe must fail (proving an exclusive holder),
    then LOCK_EX on the inherited open-file description must succeed. Neither
    operation unlocks the inherited description. The worker keeps it open until
    process exit, even when its supervisor closes its own copy or dies.
    """
    import fcntl
    if lock_fd is None:
        value = os.environ.get(GPU_LOCK_FD_ENV, "")
        require(value.isascii() and value.isdecimal(), "inherited GPU lock FD required; use the supervising CLI")
        lock_fd = int(value)
    require(type(lock_fd) is int and lock_fd >= 3, "invalid inherited GPU lock FD")
    try:
        _same_lock_inode(lock_fd, GPU_LOCK)
        with _lock_file(GPU_LOCK) as probe:
            _same_lock_inode(probe.fileno(), GPU_LOCK)
            try:
                fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("GPU lock FD is not protected by an exclusive flock")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError("GPU lock FD does not own the machine flock") from e
        _same_lock_inode(lock_fd, GPU_LOCK)
    except OSError as e:
        raise ValueError("invalid inherited GPU lock FD or machine lock file") from e
    return lock_fd


@contextlib.contextmanager
def gpu_lock(path=None):
    import fcntl
    path = Path(GPU_LOCK if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_file(path) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"active GPU workflow holds {path}") from e
        _same_lock_inode(f.fileno(), path)
        # Close only: LOCK_UN would also release surviving workers' inherited lock.
        # Keep the same inode permanently; workers inherit this descriptor.
        yield f.fileno()


def check_idle_gpu():
    processes = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True, timeout=10)
    active = []
    pattern = re.compile(r"(?:^|[/\s])(?:train(?:_rgbd)?|run_colab_hard|run_colab_[\w]+|eval|record_dp|rollout_logger|run_generalization|build_repro_package)\.py(?:\s|$)")
    for line in processes.stdout.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != os.getpid() and pattern.search(command):
            active.append(int(pid))
    require(not active, f"other training/evaluation processes active: {active}")
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True, timeout=10)
    require(all(x.strip().isdigit() for x in gpu.stdout.splitlines() if x.strip()), "unrecognized GPU process response")
    pids = [int(x.strip()) for x in gpu.stdout.splitlines() if x.strip().isdigit()]
    require(not pids, f"GPU already has compute clients: {pids}; run after Hard and release prior CUDA clients")
    return {"other_rollout_pids": active, "gpu_compute_pids": pids, "lock": str(GPU_LOCK)}


def supervised(command, out, timeout, *, cwd=ROOT, lock_fd=None):
    """One owned process group, bounded wall time, durable progress every <=30 s."""
    require(0 < timeout <= 43200, "timeout must be 1..43200 seconds")
    if lock_fd is not None:
        require_gpu_lock(lock_fd)
    out = Path(out)
    start = time.monotonic()
    with (out / "worker.log").open("w") as log:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        # Do not let a stale caller environment authorize a CPU-only child.
        env.pop(GPU_LOCK_FD_ENV, None)
        if lock_fd is not None:
            env[GPU_LOCK_FD_ENV] = str(lock_fd)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             pass_fds=(() if lock_fd is None else (lock_fd,)))
        def interrupted(signum, frame):
            raise RuntimeError(f"supervisor interrupted by signal {signum}")

        previous_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            while True:
                elapsed = time.monotonic() - start
                write_json(out / "progress.json", {"pid": p.pid, "elapsed_seconds": round(elapsed, 1), "status": "running"})
                print(f"[validation] {out.name} pid={p.pid} elapsed={elapsed:.0f}s", flush=True)
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError(f"worker exceeded {timeout}s")
                try:
                    rc = p.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(rc == 0, f"worker exit {rc}; inspect {out / 'worker.log'}")
            write_json(out / "progress.json", {"pid": p.pid, "status": "completed", "elapsed_seconds": round(time.monotonic() - start, 1)})
        except BaseException:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            # Also reap lingering descendants if the worker exited before them.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=10)
            write_json(out / "progress.json", {"pid": p.pid, "status": "failed", "elapsed_seconds": round(time.monotonic() - start, 1)})
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def seal_files(root):
    root = Path(root)
    manifest = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
                if p.is_file() and p != root / "ARTIFACT_SHA256.json"}
    write_json(root / "ARTIFACT_SHA256.json", manifest)
    return manifest


def verify_seal(root):
    root = Path(root)
    manifest = json.loads((root / "ARTIFACT_SHA256.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "ARTIFACT_SHA256.json"}
    require(actual == set(manifest), "archive has missing or extra files")
    for name, digest in manifest.items():
        path = root / safe_relative(name)
        require(not path.is_symlink() and sha256(path) == digest, f"archive SHA mismatch: {name}")
    return len(manifest)
