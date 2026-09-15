"""Medium-only Colab supervisor. Trains from scratch, validates, and never submits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


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
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def run_stage(cmd, log, timeout, update, env, cwd):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
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
        try:
            rc = p.wait(timeout=timeout)
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


def _eval_cfg(path, n, seed0=5000):
    seeds = list(range(seed0, seed0 + n))
    Path(path).write_text("eval:\n  n_episodes: %d\n  seeds: %s\n" % (n, json.dumps(seeds)))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    repo = Path(plan["repo"])
    root = Path(plan["root"])
    py = plan["python"]
    exp = plan["exp"]
    assert repo == Path("/content/berlin-marso-hackathon")
    assert root == Path("/content/drive/MyDrive/marso")
    assert plan["level"] == "medium" and plan["method"] == "dp_rgb_medium"
    assert int(plan["max_episode_steps"]) == 500
    assert int(plan["total_iters"]) == 30000
    assert all(x not in exp.lower() for x in ("easy", "s1", "old"))
    assert exp.startswith("medium_")
    assert exp.isascii() and exp.replace("_", "").replace("-", "").isalnum()
    runroot = root / "runs" / "medium" / exp
    ckdir = root / "ckpts" / "medium" / exp
    if ckdir.exists() and any(ckdir.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty checkpoint dir: {ckdir}")
    runroot.mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)
    lockpath = repo / "outputs" / "locks" / f"{exp}.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    status = runroot / "status.json"
    state = {
        "exp": exp,
        "level": "medium",
        "pid": os.getpid(),
        "started": time.time(),
        "total_iters": plan["total_iters"],
        "status": "starting",
        "submission_executed": False,
        "git_push_executed": False,
        "automatic_hard_or_act": False,
    }

    def update(**kw):
        state.update(kw)
        state["updated"] = time.time()
        save_json(status, state)

    with lockpath.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Same medium experiment is already running")
        try:
            for rel, expected in plan["source_sha256"].items():
                assert sha(repo / rel) == expected, f"Source changed: {rel}"
            save_json(runroot / "plan.json", plan)
            env = dict(
                os.environ,
                DISPLAY="",
                PYOPENGL_PLATFORM="egl",
                HDF5_USE_FILE_LOCKING="FALSE",
                PYTHONUNBUFFERED="1",
                PYTHONPATH=str(repo),
                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            )
            train = [
                py,
                "il/train.py",
                "method=dp_rgb_medium",
                f"flags.exp_name={exp}",
                "flags.total_iters=30000",
                f"flags.ckpt_dir={ckdir}",
                "flags.save_freq=1000",
                "flags.eval_freq=5000",
                "flags.num_eval_episodes=32",
                "flags.num_eval_envs=8",
                "flags.skip_initial_eval=true",
            ]
            update(status="training")
            run_stage(train, runroot / "train.log", 9000, update, env, repo)
            import torch

            final = ckdir / "final.pt"
            ck = torch.load(final, map_location="cpu", weights_only=False)
            assert int(ck["iteration"]) == 30000
            del ck
            best = ckdir / "best_eval_sort_accuracy.pt"
            if not best.exists():
                best = final
            best_hash = sha(best)
            update(status="sweeping", checkpoint=str(best), checkpoint_sha256=best_hash)
            cfg32 = runroot / "eval32.yaml"
            _eval_cfg(cfg32, 32)
            results = []
            for horizon, steps in ((8, 16), (4, 16), (8, 32), (4, 32)):
                label = f"eval_h{horizon}_d{steps}"
                out = runroot / f"{label}.jsonl"
                cmd = [
                    py,
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
                run_stage(cmd, runroot / f"{label}.log", 1800, update, env, repo)
                rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
                assert len(rows) == 1 and rows[0]["n_episodes"] == 32 and rows[0]["requested_n_episodes"] == 32
                assert rows[0]["level"] == "medium" and rows[0]["max_episode_steps"] == 500
                assert sha(best) == best_hash
                results.append(rows[0])
                save_json(runroot / "evaluation_results.json", results)
            winner = max(results, key=lambda r: r.get("sort_accuracy", 0.0))
            opts = [f"+policy_kwargs.{k}={v}" for k, v in winner["policy_kwargs"].items()]
            update(status="recording_preview")
            cfg1 = runroot / "eval1.yaml"
            _eval_cfg(cfg1, 1)
            run_stage(
                [
                    py,
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
                repo,
            )
            videos = sorted((runroot / "preview" / "videos").glob("*.mp4"))
            assert videos, "Preview did not produce an MP4"
            update(status="diagnostics")
            cfg8 = runroot / "diag8.yaml"
            _eval_cfg(cfg8, 8)
            diag = runroot / "diagnostics.json"
            run_stage(
                [
                    py,
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
                repo,
            )
            assert diag.is_file(), "Diagnostics file missing"
            summary = {
                "status": "completed",
                "exp": exp,
                "level": "medium",
                "iterations": 30000,
                "checkpoint": str(best),
                "checkpoint_sha256": best_hash,
                "best_eval": winner,
                "eval_runs": results,
                "videos": [{"path": str(v), "sha256": sha(v)} for v in videos],
                "diagnostics": str(diag),
                "source_sha256": plan["source_sha256"],
                "submission_executed": False,
                "git_push_executed": False,
                "automatic_hard_or_act": False,
            }
            save_json(runroot / "summary.json", summary)
            update(status="completed", summary=str(runroot / "summary.json"), finished=time.time())
        except Exception as exc:
            update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())
            raise


if __name__ == "__main__":
    main()
