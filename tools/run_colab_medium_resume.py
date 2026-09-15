"""Scoped medium resume supervisor for Colab.

This resumes one validated medium Diffusion Policy run from local checkpoint copies, writes all
training/eval artifacts under a local experiment root first, and mirrors additively to Drive.
"""
from __future__ import annotations

import argparse
import fcntl
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


EXPECTED_REPO = Path("/content/berlin-marso-hackathon")
EXPECTED_PYTHON = "/content/marso-py312/bin/python"
EXPECTED_DRIVE_ROOT = Path("/content/drive/MyDrive/marso")
EXPECTED_LOCAL_BASE = Path("/content/marso-resume")
EXPECTED_RESUME_SHA256 = "8082cbb5b4da6a871ad754266d13a8efad714790e6b956502f8f440b328a205b"
EXPECTED_PRIOR_BEST_SHA256 = "95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb"
EXPECTED_RESUME_ITERATION = 29000
EXPECTED_PRIOR_BEST_ITERATION = 25000
EXPECTED_PRIOR_BEST_SORT_ACCURACY = 1.0
EXPECTED_PRIOR_EXP = "medium_resume_20260914-040647"
EXPECTED_DRIVE_INPUT_SOURCE = f"gdrive:marso/recoveries/medium/{EXPECTED_PRIOR_EXP}/ckpts"
MEDIUM_DEMO_MARKER = Path("demos/medium/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
SAFE_EXP_RE = re.compile(r"^medium_resume_[A-Za-z0-9_-]+$")

MEDIUM_PRESET = {
    "level": "medium",
    "method": "dp_rgb_medium",
    "total_iters": 30000,
    "max_episode_steps": 500,
    "num_eval_episodes": 32,
    "num_eval_envs": 8,
    "save_freq": 1000,
    "eval_freq": 5000,
    "batch_size": 128,
    "lr": 1e-4,
    "obs_horizon": 2,
    "act_horizon": 8,
    "pred_horizon": 16,
    "num_diffusion_iters": 100,
    "eval_inference_steps": 16,
    "visual_encoder": "resnet18",
    "num_kp": 32,
    "obs_mode": "rgb",
    "obs_camera": "scene",
    "clip_actions": True,
    "image_aug_pad": 4,
    "proprio_noise_std": 0.0,
    "amp": True,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def _safe_exp(exp):
    return (isinstance(exp, str) and exp.isascii() and SAFE_EXP_RE.fullmatch(exp) is not None
            and exp != EXPECTED_PRIOR_EXP)


def _as_path(value, key):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return Path(value)


def validate_input_identity(plan):
    for key, expected in (
        ("prior_exp", EXPECTED_PRIOR_EXP),
        ("drive_input_source", EXPECTED_DRIVE_INPUT_SOURCE),
        ("resume_sha256", EXPECTED_RESUME_SHA256),
        ("prior_best_sha256", EXPECTED_PRIOR_BEST_SHA256),
    ):
        if plan.get(key) != expected:
            raise ValueError(f"{key} must match the verified recovery input: {expected}")


def load_plan(path):
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text())
    exp = plan.get("exp")
    if not _safe_exp(exp):
        raise ValueError(f"Unsafe experiment name: {exp!r}")
    if _as_path(plan.get("repo"), "repo") != EXPECTED_REPO:
        raise ValueError(f"repo must be {EXPECTED_REPO}")
    if plan.get("python") != EXPECTED_PYTHON:
        raise ValueError(f"python must be {EXPECTED_PYTHON}")
    if _as_path(plan.get("drive_root"), "drive_root") != EXPECTED_DRIVE_ROOT:
        raise ValueError(f"drive_root must be {EXPECTED_DRIVE_ROOT}")
    local_root = _as_path(plan.get("local_root"), "local_root")
    if local_root != EXPECTED_LOCAL_BASE / exp:
        raise ValueError(f"local_root must be {EXPECTED_LOCAL_BASE / exp}")
    validate_input_identity(plan)
    for key, expected in (
        ("level", "medium"),
        ("method", "dp_rgb_medium"),
        ("total_iters", 30000),
        ("max_episode_steps", 500),
    ):
        if plan.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    resume_checkpoint = _as_path(plan.get("resume_checkpoint"), "resume_checkpoint")
    prior_best_checkpoint = _as_path(plan.get("prior_best_checkpoint"), "prior_best_checkpoint")
    if not _is_relative_to(resume_checkpoint, local_root):
        raise ValueError("resume_checkpoint must be the local input copy under local_root")
    if not _is_relative_to(prior_best_checkpoint, local_root):
        raise ValueError("prior_best_checkpoint must be the local input copy under local_root")
    if resume_checkpoint.name != "latest.pt" or prior_best_checkpoint.name != "best_eval_sort_accuracy.pt":
        raise ValueError("checkpoint input names must be latest.pt and best_eval_sort_accuracy.pt")
    if not isinstance(plan.get("source_sha256"), dict) or not plan["source_sha256"]:
        raise ValueError("source_sha256 must be a non-empty map")
    return plan


def _is_relative_to(path, root):
    try:
        Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        return True
    except ValueError:
        return False


def verify_source_sha(repo, source_sha256):
    for rel, expected in source_sha256.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"Unsafe source path: {rel}")
        actual = sha(Path(repo) / rel_path)
        if actual != expected:
            raise RuntimeError(f"Source changed: {rel}")


def _check_empty_or_allowed(root, allowed_paths):
    root = Path(root)
    allowed = {Path(p).resolve(strict=False) for p in allowed_paths}
    if not root.exists():
        return
    for path in root.rglob("*"):
        resolved = path.resolve(strict=False)
        if resolved in allowed:
            continue
        if path.is_dir() and any(_is_relative_to(a, resolved) for a in allowed):
            continue
        if path.name.endswith(".tmp") and path.parent.resolve(strict=False) == root.resolve(strict=False):
            continue
        raise FileExistsError(f"Refusing to reuse non-empty target: {root}")


def _check_empty_or_plan_only(root, allowed_plan):
    _check_empty_or_allowed(root, [allowed_plan])


def ensure_remote_ready(drive_root, mount_point=Path("/content/drive"), marker_rel=MEDIUM_DEMO_MARKER, require_mount=True):
    mount_point = Path(mount_point)
    drive_root = Path(drive_root)
    if require_mount and not mount_point.is_mount():
        raise RuntimeError(f"Drive is not mounted at {mount_point}")
    if not drive_root.is_dir():
        raise RuntimeError(f"Drive root missing: {drive_root}")
    marker = drive_root / marker_rel
    if not marker.is_file():
        raise RuntimeError(f"Drive marker missing: {marker}")


def recovery_remote_root(plan):
    return Path(plan["drive_root"]) / "recoveries" / "medium" / plan["exp"]


def prepare_roots(plan, plan_path):
    local_root = Path(plan["local_root"])
    remote_root = recovery_remote_root(plan)
    _check_empty_or_allowed(local_root, [plan_path, plan["resume_checkpoint"], plan["prior_best_checkpoint"]])
    ensure_remote_ready(Path(plan["drive_root"]))
    _check_empty_or_plan_only(remote_root, plan_path)
    (local_root / "run").mkdir(parents=True, exist_ok=False)
    (local_root / "ckpts").mkdir(parents=True, exist_ok=False)
    remote_root.mkdir(parents=True, exist_ok=False)
    return local_root, remote_root


def _get_nested(obj, key):
    if not isinstance(obj, dict):
        return None
    return obj.get(key)


def _float_equal(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def validate_medium_checkpoint(ck, *, expected_iteration):
    if int(ck.get("iteration", -1)) != int(expected_iteration):
        raise AssertionError(f"checkpoint iteration must be {expected_iteration}")
    cfg = ck.get("config", {})
    args = ck.get("args", {})
    checks = {
        "max_episode_steps": 500,
        "batch_size": 128,
        "obs_horizon": 2,
        "act_horizon": 8,
        "pred_horizon": 16,
        "num_diffusion_iters": 100,
        "eval_inference_steps": 16,
        "visual_encoder": "resnet18",
        "num_kp": 32,
        "obs_mode": "rgb",
        "obs_camera": "scene",
        "clip_actions": True,
        "image_aug_pad": 4,
        "proprio_noise_std": 0.0,
        "amp": True,
    }
    for key, expected in checks.items():
        value = _get_nested(args, key)
        if value is None:
            value = _get_nested(cfg, key)
        if isinstance(expected, float):
            ok = _float_equal(value, expected)
        else:
            ok = value == expected
        if not ok:
            raise AssertionError(f"{key} mismatch: {value!r} != {expected!r}")
    if not _float_equal(args.get("lr", cfg.get("lr")), 1e-4):
        raise AssertionError("lr mismatch")
    if int(args.get("total_iters", cfg.get("total_iters", -1))) != 30000:
        raise AssertionError("total schedule mismatch")
    demo_paths = cfg.get("demo_paths") or args.get("demo_path") or []
    if isinstance(demo_paths, str):
        demo_paths = [demo_paths]
    if not any("/medium/" in str(p) or "\\medium\\" in str(p) for p in demo_paths):
        raise AssertionError("checkpoint is not sourced from medium demos")
    if any("/easy/" in str(p) or "/hard/" in str(p) for p in demo_paths):
        raise AssertionError("checkpoint mixes another level")
    validate_resume_state(ck, expected_iteration=expected_iteration)


def validate_resume_state(ck, *, expected_iteration):
    import torch

    required = {"agent", "ema_agent", "optimizer", "lr_scheduler", "ema_state", "scaler",
                "rng", "config", "args", "best_eval_metrics", "eval_history"}
    if not required.issubset(ck):
        raise AssertionError(f"Missing resume state: {sorted(required - ck.keys())}")
    for section in ("agent", "ema_agent"):
        state = ck[section]
        if not isinstance(state, dict) or not state:
            raise AssertionError(f"Empty model state: {section}")
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item():
                raise AssertionError(f"non-finite or invalid tensor in {section}.{name}")
    optimizer = ck["optimizer"]
    if not isinstance(optimizer, dict) or len(optimizer.get("state", {})) != 197 or not optimizer.get("param_groups"):
        raise AssertionError("optimizer must contain 197 state entries and parameter groups")
    scheduler = ck["lr_scheduler"]
    # The trainer labels final saves as total_iters after stepping at total_iters - 1.
    # Intermediate saves keep the zero-based iteration and require iteration + 1.
    expected_scheduler_step = (
        expected_iteration if expected_iteration == MEDIUM_PRESET["total_iters"] else expected_iteration + 1
    )
    if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != expected_scheduler_step:
        raise AssertionError("lr_scheduler step mismatch")
    ema = ck["ema_state"]
    if not isinstance(ema, dict) or not ema.get("shadow_params"):
        raise AssertionError("Missing EMA resume state")
    if not all(torch.is_tensor(t) and torch.isfinite(t).all().item() for t in ema["shadow_params"]):
        raise AssertionError("non-finite EMA resume state")
    rng = ck["rng"]
    if not isinstance(rng, dict) or any(rng.get(k) is None for k in ("python", "numpy", "torch", "cuda")):
        raise AssertionError("Missing Python/numpy/torch/CUDA RNG state")
    if not isinstance(rng["python"], tuple) or not rng["python"] or not isinstance(rng["numpy"], tuple) or not rng["numpy"]:
        raise AssertionError("Invalid Python/numpy RNG state")
    if not isinstance(rng["cuda"], (list, tuple)) or not rng["cuda"]:
        raise AssertionError("Missing CUDA RNG state")
    for state in [rng["torch"], *rng["cuda"]]:
        if not torch.is_tensor(state) or state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1 or not state.numel():
            raise AssertionError("Invalid torch/CUDA RNG state")


def validate_prior_best_checkpoint(ck):
    validate_medium_checkpoint(ck, expected_iteration=EXPECTED_PRIOR_BEST_ITERATION)
    best = ck.get("best_eval_metrics", {}).get("sort_accuracy")
    if not _float_equal(best, EXPECTED_PRIOR_BEST_SORT_ACCURACY):
        raise AssertionError("prior best metric mismatch")


def load_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def verify_and_seed_inputs(plan):
    validate_input_identity(plan)
    resume = Path(plan["resume_checkpoint"])
    prior = Path(plan["prior_best_checkpoint"])
    if sha(resume) != EXPECTED_RESUME_SHA256:
        raise AssertionError("resume checkpoint sha256 mismatch")
    if sha(prior) != EXPECTED_PRIOR_BEST_SHA256:
        raise AssertionError("prior best checkpoint sha256 mismatch")
    resume_ck = load_checkpoint(resume)
    validate_medium_checkpoint(resume_ck, expected_iteration=EXPECTED_RESUME_ITERATION)
    prior_ck = load_checkpoint(prior)
    validate_prior_best_checkpoint(prior_ck)
    ckpts = Path(plan["local_root"]) / "ckpts"
    seeded_best = ckpts / "best_eval_sort_accuracy.pt"
    atomic_copy_verified(prior, seeded_best)
    lineage = {
        "prior_exp": plan["prior_exp"],
        "drive_input_source": plan["drive_input_source"],
        "resume_checkpoint": str(resume),
        "resume_sha256": plan["resume_sha256"],
        "resume_iteration": EXPECTED_RESUME_ITERATION,
        "prior_best_checkpoint": str(prior),
        "prior_best_sha256": plan["prior_best_sha256"],
        "prior_best_iteration": EXPECTED_PRIOR_BEST_ITERATION,
        "prior_best_sort_accuracy": EXPECTED_PRIOR_BEST_SORT_ACCURACY,
        "seeded_best": str(seeded_best),
        "seeded_best_sha256": sha(seeded_best),
    }
    save_json(Path(plan["local_root"]) / "run" / "input_lineage.json", lineage)
    return lineage


def build_train_command(plan):
    local_root = Path(plan["local_root"])
    exp = plan["exp"]
    return [
        plan["python"],
        "il/train.py",
        "method=dp_rgb_medium",
        f"flags.resume={Path(plan['resume_checkpoint'])}",
        f"flags.ckpt_dir={local_root / 'ckpts'}",
        f"flags.exp_name={exp}",
        "flags.total_iters=30000",
        "flags.save_freq=1000",
        "flags.eval_freq=5000",
        "flags.num_eval_episodes=32",
        "flags.num_eval_envs=8",
        "flags.skip_initial_eval=true",
    ]


def _eval_cfg(path, n, seed0=5000):
    seeds = list(range(seed0, seed0 + n))
    Path(path).write_text("eval:\n  n_episodes: %d\n  seeds: %s\n" % (n, json.dumps(seeds)))
    return seeds


def _iter_mirror_files(local_root):
    local_root = Path(local_root)
    for path in sorted(local_root.rglob("*")):
        if any(part in (".sync", "inputs") for part in path.relative_to(local_root).parts):
            continue
        if path.name == "durability.json":
            continue
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.endswith(".tmp") or path.name.endswith(".lock") or ".tmp" in path.name:
            continue
        yield path


def _file_meta(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _safe_remote_dest(remote_root, rel):
    rel = Path(rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe relative mirror path: {rel}")
    dest = Path(remote_root) / rel
    if not _is_relative_to(dest, remote_root):
        raise ValueError(f"remote write escaped recovery root: {dest}")
    return dest


def atomic_copy_verified(src, dest, guard=None):
    src = Path(src)
    dest = Path(dest)
    if guard is not None:
        guard()
    dest.parent.mkdir(parents=True, exist_ok=True)
    before_meta = _file_meta(src)
    before_hash = sha(src)
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(src, tmp)
        after_meta = _file_meta(src)
        if after_meta != before_meta:
            tmp.unlink(missing_ok=True)
            if src.suffix == ".log":
                return {"status": "skipped_growing_log", "sha256": before_hash}
            raise RuntimeError(f"source changed during copy: {src}")
        copied_hash = sha(tmp)
        if copied_hash != before_hash:
            raise RuntimeError(f"copy hash mismatch: {src}")
        if guard is not None:
            guard()
        os.replace(tmp, dest)
        if sha(dest) != before_hash:
            raise RuntimeError(f"remote verify mismatch: {dest}")
        return {"status": "copied", "sha256": before_hash}
    finally:
        tmp.unlink(missing_ok=True)


def _drive_guard(local_root, remote_root):
    local_root, remote_root = Path(local_root), Path(remote_root)
    if not _safe_exp(local_root.name) or local_root != EXPECTED_LOCAL_BASE / local_root.name:
        raise ValueError("Unexpected local mirror scope")
    if remote_root != EXPECTED_DRIVE_ROOT / "recoveries" / "medium" / local_root.name:
        raise ValueError("Unexpected remote mirror scope")
    ensure_remote_ready(EXPECTED_DRIVE_ROOT)


def sync_tree_once(local_root, remote_root, cache=None, final=False):
    local_root = Path(local_root)
    remote_root = Path(remote_root)
    _drive_guard(local_root, remote_root)
    guard = lambda: _drive_guard(local_root, remote_root)
    cache = dict(cache or {})
    copied = []
    skipped = []
    errors = []
    for src in _iter_mirror_files(local_root):
        rel = src.relative_to(local_root)
        dest = _safe_remote_dest(remote_root, rel)
        meta = _file_meta(src)
        cache_key = str(rel)
        old = cache.get(cache_key, {})
        if not final and old.get("meta") == meta and dest.is_file() and _file_meta(dest)["size"] == meta["size"]:
            skipped.append(cache_key)
            continue
        try:
            result = atomic_copy_verified(src, dest, guard=guard)
            if result["status"] == "copied":
                copied.append(cache_key)
                cache[cache_key] = {"meta": meta, "sha256": result["sha256"]}
            else:
                skipped.append(cache_key)
                if final:
                    errors.append({"path": cache_key, "error": "File is still growing"})
        except Exception as exc:
            errors.append({"path": cache_key, "error": f"{type(exc).__name__}: {exc}"})
    ok = not errors
    return {"ok": ok, "copied": copied, "skipped": skipped, "errors": errors, "cache": cache}


def run_sync_worker(local_root, remote_root, cache, *, final=False, timeout=20):
    req_dir = Path(local_root) / "run" / ".sync"
    req_dir.mkdir(parents=True, exist_ok=True)
    token = f"{int(time.time() * 1000)}-{os.getpid()}"
    req = req_dir / f"{token}.request.json"
    resp = req_dir / f"{token}.response.json"
    save_json(req, {"local_root": str(local_root), "remote_root": str(remote_root), "cache": cache, "final": final})
    cmd = [sys.executable, __file__, "--sync-worker", str(req), str(resp)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if r.returncode != 0:
            return {"ok": False, "errors": [{"path": ".", "error": (r.stderr or r.stdout).strip()}], "cache": cache}
        return json.loads(resp.read_text())
    except subprocess.TimeoutExpired:
        return {"ok": False, "errors": [{"path": ".", "error": f"sync worker timed out after {timeout}s"}], "cache": cache}
    except Exception as exc:
        return {"ok": False, "errors": [{"path": ".", "error": str(exc)}], "cache": cache}
    finally:
        req.unlink(missing_ok=True)
        resp.unlink(missing_ok=True)


def run_stage(cmd, log, timeout, update, env, cwd, sync=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[medium-resume] starting {' '.join(str(x) for x in cmd)}", flush=True)
    with log.open("a") as f:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        update(stage_pid=p.pid, stage_started=time.time(), log=str(log), command=[str(x) for x in cmd])
        deadline = time.time() + timeout
        next_progress = 0.0
        next_sync = 0.0
        try:
            while True:
                rc = p.poll()
                now = time.time()
                if rc is not None:
                    break
                if now >= deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if now >= next_progress:
                    update(stage_pid=p.pid, stage_elapsed_s=round(now - (deadline - timeout), 1))
                    with log.open("rb") as progress_file:
                        progress_file.seek(max(0, log.stat().st_size - 2400))
                        lines = progress_file.read().decode(errors="replace").replace("\r", "\n").splitlines()
                    progress = next((line.strip() for line in reversed(lines) if line.strip()), "initializing")
                    update(latest_progress=progress[-350:])
                    print(f"[medium-resume] pid={p.pid} elapsed={now - (deadline - timeout):.0f}s {progress[-350:]}", flush=True)
                    next_progress = now + 60
                if sync is not None and now >= next_sync:
                    sync(final=False)
                    next_sync = now + 30
                time.sleep(5)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise RuntimeError(f"Time limit reached; logs and checkpoints retained: {log}")
    if rc:
        raise RuntimeError(f"Process exited {rc}; see {log}")
    update(stage_pid=None)
    if sync is not None:
        sync(final=False)


def assert_final_checkpoint(path):
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("iteration", -1)) != 30000:
        raise AssertionError("final checkpoint iteration is not 30000")
    validate_medium_checkpoint(ck, expected_iteration=30000)
    for section in ("agent", "ema_agent"):
        state = ck.get(section, {})
        if not state:
            raise AssertionError(f"Empty final model state: {section}")
        for name, tensor in state.items():
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                if not torch.isfinite(tensor).all().item():
                    raise AssertionError(f"non-finite tensor in {section}.{name}")


def _summary_files(root):
    return [{"path": str(p), "sha256": sha(p)} for p in sorted(Path(root).glob("*")) if p.is_file() and p.name not in ("status.json", "durability.json")]


def run_pipeline(plan, plan_path):
    verify_source_sha(plan["repo"], plan["source_sha256"])
    local_root, remote_root = prepare_roots(plan, plan_path)
    runroot = local_root / "run"
    ckdir = local_root / "ckpts"
    lockpath = local_root / f"{plan['exp']}.lock"
    status_path = runroot / "status.json"
    state = {
        "exp": plan["exp"],
        "pipeline_status": "starting",
        "remote_status": "not_synced",
        "remote_verified": False,
        "started": time.time(),
        "submission_executed": False,
        "git_push_executed": False,
        "automatic_hard_or_act": False,
    }
    sync_cache = {}

    def update(**kw):
        state.update(kw)
        state["updated"] = time.time()
        if state.get("pipeline_status") == "completed" and not state.get("remote_verified"):
            state["status"] = "sync_pending"
        else:
            state["status"] = state.get("pipeline_status")
        save_json(status_path, state)

    def sync(final=False):
        nonlocal sync_cache
        result = run_sync_worker(local_root, remote_root, sync_cache, final=final, timeout=180 if final else 120)
        sync_cache = result.get("cache", sync_cache)
        if result.get("ok"):
            state["remote_status"] = "verified" if final else "synced"
            state["last_sync"] = {"time": time.time(), "copied": result.get("copied", [])}
        else:
            state["remote_status"] = "sync_pending"
            state["last_sync_error"] = result.get("errors", [])
        state["remote_verified"] = bool(final and result.get("ok"))
        update()
        return result

    env = dict(
        os.environ,
        DISPLAY="",
        PYOPENGL_PLATFORM="egl",
        HDF5_USE_FILE_LOCKING="FALSE",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(plan["repo"]),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    with lockpath.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Same medium resume experiment is already running")
        try:
            save_json(runroot / "plan.json", plan)
            update(pipeline_status="validating_inputs")
            lineage = verify_and_seed_inputs(plan)
            sync(final=False)
            update(pipeline_status="training")
            run_stage(build_train_command(plan), runroot / "train.log", 5400, update, env, plan["repo"], sync=sync)
            final = ckdir / "final.pt"
            assert_final_checkpoint(final)
            best = ckdir / "best_eval_sort_accuracy.pt"
            if not best.exists():
                best = final
            best_hash = sha(best)
            update(pipeline_status="sweeping", checkpoint=str(best), checkpoint_sha256=best_hash)
            cfg32 = runroot / "eval32.yaml"
            _eval_cfg(cfg32, 32)
            results = []
            for horizon, steps in ((8, 16), (4, 16), (8, 32), (4, 32)):
                label = f"eval_h{horizon}_d{steps}"
                out = runroot / f"{label}.jsonl"
                cmd = [
                    plan["python"],
                    "eval.py",
                    "difficulty=medium",
                    "obs_mode=rgb",
                    "policy=warehouse_sort.il_policy:load_dp_rgb",
                    f"checkpoint={best}",
                    f"eval_config={cfg32}",
                    "record_video=false",
                    f"results_file={out}",
                    f"+policy_kwargs.act_horizon={horizon}",
                    f"+policy_kwargs.num_inference_steps={steps}",
                ]
                update(evaluation=label)
                run_stage(cmd, runroot / f"{label}.log", 1800, update, env, plan["repo"], sync=sync)
                rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
                assert len(rows) == 1 and rows[0]["n_episodes"] == 32 and rows[0]["requested_n_episodes"] == 32
                assert rows[0]["level"] == "medium" and rows[0]["max_episode_steps"] == 500
                assert sha(best) == best_hash
                results.append(rows[0])
                save_json(runroot / "evaluation_results.json", results)
            winner = max(results, key=lambda r: r.get("sort_accuracy", 0.0))
            opts = [f"+policy_kwargs.{k}={v}" for k, v in winner["policy_kwargs"].items()]
            update(pipeline_status="recording_preview")
            cfg1 = runroot / "eval1.yaml"
            _eval_cfg(cfg1, 1)
            run_stage(
                [
                    plan["python"],
                    "eval.py",
                    "difficulty=medium",
                    "obs_mode=rgb",
                    "policy=warehouse_sort.il_policy:load_dp_rgb",
                    f"checkpoint={best}",
                    f"eval_config={cfg1}",
                    "record_video=true",
                    "video_envs=1",
                    f"hydra.run.dir={runroot / 'preview'}",
                ]
                + opts,
                runroot / "preview.log",
                1200,
                update,
                env,
                plan["repo"],
                sync=sync,
            )
            videos = sorted((runroot / "preview" / "videos").glob("*.mp4"))
            assert videos, "Preview did not produce an MP4"
            update(pipeline_status="diagnostics")
            cfg8 = runroot / "diag8.yaml"
            _eval_cfg(cfg8, 8)
            diag = runroot / "diagnostics.json"
            run_stage(
                [
                    plan["python"],
                    "tools/rollout_logger.py",
                    "difficulty=medium",
                    "obs_mode=rgb",
                    "policy=warehouse_sort.il_policy:load_dp_rgb",
                    f"checkpoint={best}",
                    f"eval_config={cfg8}",
                    "num_envs=8",
                    "+log_episodes=8",
                    f"+log_out={diag}",
                ]
                + opts,
                runroot / "diagnostics.log",
                1200,
                update,
                env,
                plan["repo"],
                sync=sync,
            )
            assert diag.is_file(), "Diagnostics file missing"
            summary = {
                "pipeline_status": "completed",
                "durability_receipt": "run/durability.json",
                "exp": plan["exp"],
                "level": "medium",
                "iterations": 30000,
                "local_root": str(local_root),
                "remote_root": str(remote_root),
                "checkpoint": str(best),
                "checkpoint_sha256": best_hash,
                "input_lineage": lineage,
                "best_eval": winner,
                "eval_runs": results,
                "videos": [{"path": str(v), "sha256": sha(v)} for v in videos],
                "diagnostics": {"path": str(diag), "sha256": sha(diag)},
                "run_files": _summary_files(runroot),
                "ckpt_files": _summary_files(ckdir),
                "source_sha256": plan["source_sha256"],
                "submission_executed": False,
                "git_push_executed": False,
                "automatic_hard_or_act": False,
            }
            save_json(runroot / "summary.json", summary)
            update(pipeline_status="completed", summary=str(runroot / "summary.json"), finished=time.time())
            final_sync = sync(final=True)
            if not final_sync.get("ok"):
                print("[medium-resume] local pipeline completed; Drive verification pending", flush=True)
            else:
                print("[medium-resume] COMPLETED: artifact hashes verified on Drive; see run/durability.json", flush=True)
        except Exception as exc:
            update(pipeline_status="failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
            sync(final=False)
            raise


def finalize_durability(local_root, remote_root, result):
    """Commit a receipt only after every immutable payload is verified remotely."""
    if not result.get("ok"):
        raise RuntimeError("Cannot certify a failed sync")
    local_root, remote_root = Path(local_root), Path(remote_root)
    guard = lambda: _drive_guard(local_root, remote_root)
    guard()
    manifest = {}
    for src in _iter_mirror_files(local_root):
        rel = str(src.relative_to(local_root))
        if rel == "run/status.json":
            continue  # live operational state is not an immutable payload
        entry = result["cache"].get(rel)
        if not entry or entry["meta"] != _file_meta(src):
            raise RuntimeError(f"Final source not stable: {rel}")
        dest = _safe_remote_dest(remote_root, rel)
        if sha(dest) != entry["sha256"]:
            raise RuntimeError(f"Final remote hash differs: {rel}")
        manifest[rel] = {"sha256": entry["sha256"], "size": entry["meta"]["size"]}
    required = {"run/summary.json", "ckpts/final.pt", "run/diagnostics.json", "run/evaluation_results.json"}
    if not required.issubset(manifest):
        raise RuntimeError("Missing final recovery artifacts")
    receipt = {"remote_verified": True, "verified_at": time.time(), "files": manifest,
               "remote_root": str(remote_root)}
    path = local_root / "run" / "durability.json"
    save_json(path, receipt)
    atomic_copy_verified(path, remote_root / "run" / "durability.json", guard=guard)
    status = local_root / "run" / "status.json"
    state = json.loads(status.read_text())
    state.update(status="completed", pipeline_status="completed", remote_verified=True,
                 remote_status="verified", durability_receipt=str(path))
    save_json(status, state)
    atomic_copy_verified(status, remote_root / "run" / "status.json", guard=guard)
    return receipt


def sync_worker_main(request, response):
    req = json.loads(Path(request).read_text())
    result = sync_tree_once(req["local_root"], req["remote_root"], req.get("cache"), final=bool(req.get("final")))
    if req.get("final") and result.get("ok"):
        try:
            receipt = finalize_durability(req["local_root"], req["remote_root"], result)
            result["receipt_files"] = len(receipt["files"])
        except Exception as exc:
            result["ok"] = False
            result["errors"].append({"path": "run/durability.json", "error": str(exc)})
    save_json(response, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--sync-worker", nargs=2, metavar=("REQUEST", "RESPONSE"))
    args = ap.parse_args()
    if args.sync_worker:
        sync_worker_main(args.sync_worker[0], args.sync_worker[1])
        return
    if not args.plan:
        ap.error("plan JSON path is required")
    plan = load_plan(args.plan)
    run_pipeline(plan, Path(args.plan))


if __name__ == "__main__":
    main()
