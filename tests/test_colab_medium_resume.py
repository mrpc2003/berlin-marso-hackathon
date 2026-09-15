import ast
import copy
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.run_colab_medium_resume as resume


def _plan(exp="medium_resume_29000_20260914-070000"):
    root = f"/content/marso-resume/{exp}"
    return {
        "repo": "/content/berlin-marso-hackathon",
        "python": "/content/marso-py312/bin/python",
        "exp": exp,
        "local_root": root,
        "drive_root": "/content/drive/MyDrive/marso",
        "level": "medium",
        "method": "dp_rgb_medium",
        "total_iters": 30000,
        "max_episode_steps": 500,
        "resume_checkpoint": f"{root}/inputs/latest.pt",
        "resume_sha256": resume.EXPECTED_RESUME_SHA256,
        "prior_best_checkpoint": f"{root}/inputs/best_eval_sort_accuracy.pt",
        "prior_best_sha256": resume.EXPECTED_PRIOR_BEST_SHA256,
        "prior_exp": "medium_resume_20260914-040647",
        "drive_input_source": "gdrive:marso/recoveries/medium/medium_resume_20260914-040647/ckpts",
        "source_sha256": {"tools/run_colab_medium_resume.py": "b" * 64},
    }


def test_plan_contract_and_duplicate_rejection(tmp_path):
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert resume.load_plan(path)["exp"].startswith("medium_resume_")

    bad = dict(plan, exp="../medium_resume_bad", local_root="/content/marso-resume/../medium_resume_bad")
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Unsafe experiment"):
        resume.load_plan(path)

    root = tmp_path / "local"
    root.mkdir()
    allowed = root / "plan.json"
    allowed.write_text("{}")
    resume._check_empty_or_plan_only(root, allowed)
    (root / "other.txt").write_text("occupied")
    with pytest.raises(FileExistsError):
        resume._check_empty_or_plan_only(root, allowed)

    inputs = tmp_path / "inputs-ok"
    (inputs / "inputs").mkdir(parents=True)
    plan_file = inputs / "plan.json"
    latest = inputs / "inputs" / "latest.pt"
    prior = inputs / "inputs" / "best_eval_sort_accuracy.pt"
    for p in (plan_file, latest, prior):
        p.write_text("ok")
    resume._check_empty_or_allowed(inputs, [plan_file, latest, prior])


def test_input_contract_and_resume_command():
    plan = _plan()
    cmd = resume.build_train_command(plan)
    assert cmd[:3] == ["/content/marso-py312/bin/python", "il/train.py", "method=dp_rgb_medium"]
    assert f"flags.resume={plan['resume_checkpoint']}" in cmd
    assert f"flags.ckpt_dir={plan['local_root']}/ckpts" in cmd
    assert f"flags.exp_name={plan['exp']}" in cmd
    for token in (
        "flags.total_iters=30000",
        "flags.save_freq=1000",
        "flags.eval_freq=5000",
        "flags.num_eval_episodes=32",
        "flags.num_eval_envs=8",
        "flags.skip_initial_eval=true",
    ):
        assert token in cmd


def _checkpoint(iteration=29000):
    return {
        "iteration": iteration,
        "args": {
            "max_episode_steps": 500,
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
            "total_iters": 30000,
        },
        "config": {"demo_paths": ["/content/berlin-marso-hackathon/il/demos/medium/trajectory.rgb.h5"]},
        "agent": {"weight": torch.ones(2)},
        "ema_agent": {"weight": torch.ones(2)},
        "optimizer": {"state": {i: {"step": torch.tensor(float(iteration))} for i in range(197)},
                      "param_groups": [{"params": list(range(197)), "lr": 1e-6}]},
        "lr_scheduler": {"last_epoch": iteration + 1},
        "ema_state": {"shadow_params": [torch.ones(2)], "optimization_step": iteration + 1},
        "scaler": {"scale": 65536.0},
        "rng": {"python": random.Random(0).getstate(), "numpy": np.random.RandomState(0).get_state(),
                "torch": torch.get_rng_state(), "cuda": [torch.ones(16, dtype=torch.uint8)]},
        "best_eval_metrics": {"sort_accuracy": 1.0},
        "eval_history": [],
    }


def _final_checkpoint():
    ck = _checkpoint(30000)
    # Final saves relabel the last trained iteration (29999) as total_iters.
    ck["lr_scheduler"]["last_epoch"] = 30000
    ck["ema_state"]["optimization_step"] = 30000
    return ck


@pytest.mark.parametrize("filename", ["latest.pt", "final.pt"])
def test_final_checkpoint_accepts_trainer_final_scheduler_step(tmp_path, filename):
    path = tmp_path / filename
    torch.save(_final_checkpoint(), path)
    resume.assert_final_checkpoint(path)


@pytest.mark.parametrize("step", [None, -1, 0, 25001, 29001, 29999, 30001, 30002])
def test_final_checkpoint_rejects_mismatching_scheduler_steps(tmp_path, step):
    ck = _final_checkpoint()
    ck["lr_scheduler"]["last_epoch"] = step
    path = tmp_path / "final.pt"
    torch.save(ck, path)
    with pytest.raises(AssertionError, match="scheduler step mismatch"):
        resume.assert_final_checkpoint(path)


@pytest.mark.parametrize("iteration", [25000, 29000])
@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_intermediate_input_scheduler_relation_remains_strict(iteration, offset):
    ck = _checkpoint(iteration)
    resume.validate_medium_checkpoint(ck, expected_iteration=iteration)
    ck["lr_scheduler"]["last_epoch"] = iteration + offset
    with pytest.raises(AssertionError, match="scheduler step mismatch"):
        resume.validate_medium_checkpoint(ck, expected_iteration=iteration)


def test_real_sampler_and_cpu_scheduler_reach_trainer_final_boundary(tmp_path):
    # Extract only the real sampler to avoid importing simulator/GPU dependencies.
    source = Path("il/baselines/diffusion_policy/diffusion_policy/utils.py").read_text()
    sampler_node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.ClassDef) and n.name == "IterationBasedBatchSampler")
    scope = {"Sampler": torch.utils.data.Sampler}
    exec(compile(ast.Module(body=[sampler_node], type_ignores=[]), "real-sampler", "exec"), scope)
    ck = _checkpoint()
    start_iteration = ck["iteration"] + 1
    base_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(2)), 1, False)
    sampler = scope["IterationBasedBatchSampler"](base_sampler, 30000, start_iter=start_iteration)
    assert start_iteration == 29001 and len(sampler) == 999

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    scheduler_state = scheduler.state_dict()
    scheduler_state.update(ck["lr_scheduler"])
    scheduler.load_state_dict(scheduler_state)
    iterations = []
    for iteration, _ in enumerate(sampler, start=start_iteration):
        optimizer.step()
        scheduler.step()
        iterations.append(iteration)
    assert iterations == list(range(29001, 30000))
    assert scheduler.last_epoch == 30000

    # The final checkpoint label comes from the trainer's actual assignment.
    trainer = ast.parse(Path("il/baselines/diffusion_policy/train_rgbd.py").read_text())
    final_assignment = next(n for n in ast.walk(trainer) if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "final_iter" for t in n.targets))
    final_scope = {"args": SimpleNamespace(total_iters=30000)}
    exec(compile(ast.Module(body=[final_assignment], type_ignores=[]), "trainer-final-iteration", "exec"), final_scope)
    ck = _final_checkpoint()
    ck["iteration"] = final_scope["final_iter"]
    ck["lr_scheduler"] = scheduler.state_dict()
    path = tmp_path / "final.pt"
    torch.save(ck, path)
    resume.assert_final_checkpoint(path)


def test_verified_input_constants():
    assert resume.EXPECTED_RESUME_SHA256 == "8082cbb5b4da6a871ad754266d13a8efad714790e6b956502f8f440b328a205b"
    assert resume.EXPECTED_PRIOR_BEST_SHA256 == "95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb"
    assert resume.EXPECTED_RESUME_ITERATION == 29000
    assert resume.EXPECTED_PRIOR_BEST_ITERATION == 25000
    assert resume.EXPECTED_PRIOR_BEST_SORT_ACCURACY == 1.0


@pytest.mark.parametrize("key,value", [
    ("resume_sha256", "a" * 64), ("prior_best_sha256", "a" * 64),
    ("prior_exp", "medium_20260914-014131"),
    ("drive_input_source", "gdrive:marso/ckpts/medium/medium_20260914-014131"),
    ("drive_input_source", "gdrive:marso/recoveries/medium/other/ckpts"),
    ("prior_exp", None), ("drive_input_source", None),
])
def test_plan_rejects_wrong_input_identity_even_without_load_plan(tmp_path, key, value):
    plan = dict(_plan(), **{key: value})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=key):
        resume.load_plan(path)
    with pytest.raises(ValueError, match=key):
        resume.verify_and_seed_inputs(plan)
    assert not (tmp_path / "ckpts").exists()


def test_prior_run_cannot_be_an_output_target(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(exp=resume.EXPECTED_PRIOR_EXP)))
    with pytest.raises(ValueError, match="Unsafe experiment"):
        resume.load_plan(path)
    with pytest.raises(ValueError, match="local mirror scope"):
        resume._drive_guard(resume.EXPECTED_LOCAL_BASE / resume.EXPECTED_PRIOR_EXP,
                            resume.EXPECTED_DRIVE_ROOT / "recoveries/medium" / resume.EXPECTED_PRIOR_EXP)


def test_medium_checkpoint_contract_validation():
    ck = _checkpoint()
    resume.validate_medium_checkpoint(ck, expected_iteration=29000)
    resume.validate_prior_best_checkpoint(_checkpoint(25000))
    bad = copy.deepcopy(ck)
    bad["args"]["total_iters"] = 8000
    with pytest.raises(AssertionError, match="total schedule"):
        resume.validate_medium_checkpoint(bad, expected_iteration=29000)


@pytest.mark.parametrize("iteration", [22000, 25000, 28999, 30000])
def test_rejects_wrong_resume_iteration(iteration):
    with pytest.raises(AssertionError, match="iteration must be 29000"):
        resume.validate_medium_checkpoint(_checkpoint(iteration), expected_iteration=29000)


def test_prior_best_rejects_wrong_iteration_and_metric():
    with pytest.raises(AssertionError, match="iteration must be 25000"):
        resume.validate_prior_best_checkpoint(_checkpoint(20000))
    ck = _checkpoint(25000)
    ck["best_eval_metrics"]["sort_accuracy"] = 0.8203125
    with pytest.raises(AssertionError, match="prior best metric"):
        resume.validate_prior_best_checkpoint(ck)


@pytest.mark.parametrize("section", ["agent", "ema_agent"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_input_weights_rejected(section, value):
    ck = _checkpoint()
    ck[section]["weight"][0] = value
    with pytest.raises(AssertionError, match="non-finite"):
        resume.validate_medium_checkpoint(ck, expected_iteration=29000)


@pytest.mark.parametrize("key", ["agent", "ema_agent", "optimizer", "lr_scheduler", "ema_state", "scaler", "rng"])
def test_missing_resume_state_rejected(key):
    ck = _checkpoint()
    del ck[key]
    with pytest.raises(AssertionError, match="Missing resume state"):
        resume.validate_medium_checkpoint(ck, expected_iteration=29000)


@pytest.mark.parametrize("section,value,message", [
    ("agent", {}, "Empty model state"),
    ("optimizer", {"state": {}, "param_groups": [{}]}, "197 state entries"),
    ("lr_scheduler", {"last_epoch": 29000}, "scheduler step mismatch"),
    ("ema_state", {}, "EMA resume state"),
    ("ema_state", {"shadow_params": [torch.tensor(float("nan"))]}, "non-finite EMA"),
])
def test_invalid_resume_state_rejected(section, value, message):
    ck = _checkpoint()
    ck[section] = value
    with pytest.raises(AssertionError, match=message):
        resume.validate_medium_checkpoint(ck, expected_iteration=29000)


@pytest.mark.parametrize("key", ["python", "numpy", "torch", "cuda"])
def test_missing_rng_component_rejected(key):
    ck = _checkpoint()
    del ck["rng"][key]
    with pytest.raises(AssertionError, match="RNG state"):
        resume.validate_medium_checkpoint(ck, expected_iteration=29000)


def _stage_synthetic_inputs(tmp_path, monkeypatch):
    plan = dict(_plan(), local_root=str(tmp_path))
    real_sha = resume.sha
    # Tiny serialized fixtures exercise loading/copying; only their digests stand in for Drive inputs.
    fixture_hashes = {}
    for key, filename, iteration, expected in (
        ("resume_checkpoint", "latest.pt", 29000, resume.EXPECTED_RESUME_SHA256),
        ("prior_best_checkpoint", "best_eval_sort_accuracy.pt", 25000, resume.EXPECTED_PRIOR_BEST_SHA256),
    ):
        path = tmp_path / "inputs" / filename
        path.parent.mkdir(exist_ok=True)
        torch.save(_checkpoint(iteration), path)
        fixture_hashes[real_sha(path)] = expected
        plan[key] = str(path)
    monkeypatch.setattr(resume, "sha", lambda path: fixture_hashes.get(real_sha(path), real_sha(path)))
    return plan


def test_seeded_best_and_lineage_bind_verified_inputs(tmp_path, monkeypatch):
    plan = _stage_synthetic_inputs(tmp_path, monkeypatch)
    before = {key: Path(plan[key]).read_bytes() for key in ("resume_checkpoint", "prior_best_checkpoint")}
    lineage = resume.verify_and_seed_inputs(plan)
    assert json.loads((tmp_path / "run/input_lineage.json").read_text()) == lineage
    resume.validate_input_identity(lineage)
    assert lineage["prior_exp"] == "medium_resume_20260914-040647"
    assert lineage["drive_input_source"] == "gdrive:marso/recoveries/medium/medium_resume_20260914-040647/ckpts"
    assert lineage["resume_iteration"] == 29000 and lineage["prior_best_iteration"] == 25000
    assert lineage["prior_best_sort_accuracy"] == 1.0
    assert lineage["seeded_best_sha256"] == resume.EXPECTED_PRIOR_BEST_SHA256
    assert (tmp_path / "ckpts/best_eval_sort_accuracy.pt").read_bytes() == before["prior_best_checkpoint"]
    assert all(Path(plan[key]).read_bytes() == data for key, data in before.items())
    assert not list(tmp_path.rglob("*.tmp*"))


@pytest.mark.parametrize("key", ["resume_checkpoint", "prior_best_checkpoint"])
def test_wrong_file_hash_rejected_before_loading_or_seeding(tmp_path, monkeypatch, key):
    plan = _stage_synthetic_inputs(tmp_path, monkeypatch)
    Path(plan[key]).write_bytes(b"changed input")
    monkeypatch.setattr(resume, "load_checkpoint", lambda path: pytest.fail("must hash before loading"))
    with pytest.raises(AssertionError, match="sha256 mismatch"):
        resume.verify_and_seed_inputs(plan)
    assert not (tmp_path / "ckpts").exists()


def test_drive_unmounted_no_mkdir(tmp_path):
    mount = tmp_path / "drive"
    drive_root = mount / "MyDrive" / "marso"
    with pytest.raises(RuntimeError, match="not mounted"):
        resume.ensure_remote_ready(drive_root, mount_point=mount, require_mount=True)
    assert not drive_root.exists()


def test_sync_retry_tmp_exclusion_and_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "_drive_guard", lambda *args: None)
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    (local / "run").mkdir(parents=True)
    (local / "ckpts").mkdir()
    (local / "run" / "status.json").write_text('{"pipeline_status":"training"}')
    (local / "ckpts" / "latest.pt.tmp").write_text("partial")
    result1 = resume.sync_tree_once(local, remote, final=False)
    assert result1["ok"]
    assert (remote / "run" / "status.json").read_text() == '{"pipeline_status":"training"}'
    assert not (remote / "ckpts" / "latest.pt.tmp").exists()

    (local / "ckpts" / "latest.pt").write_text("complete")
    result2 = resume.sync_tree_once(local, remote, cache=result1["cache"], final=False)
    assert result2["ok"]
    assert (remote / "ckpts" / "latest.pt").read_text() == "complete"
    assert (remote / "run" / "status.json").read_text() == '{"pipeline_status":"training"}'

    with pytest.raises(ValueError):
        resume._safe_remote_dest(remote, "../escape.txt")
    assert not (tmp_path / "escape.txt").exists()


def test_atomic_copy_failure_preserves_original_remote_file(tmp_path):
    src = tmp_path / "src.txt"
    dest = tmp_path / "remote" / "src.txt"
    dest.parent.mkdir()
    src.write_text("new")
    dest.write_text("old")
    dest.parent.chmod(0o500)
    try:
        with pytest.raises(Exception):
            resume.atomic_copy_verified(src, dest)
    finally:
        dest.parent.chmod(0o700)
    assert dest.read_text() == "old"


def test_stale_growing_log_handling(tmp_path):
    src = tmp_path / "train.log"
    dest = tmp_path / "remote" / "train.log"
    src.write_text("first")
    original_sha = resume.sha

    def growing_sha(path):
        if Path(path) == src:
            src.write_text("first\nsecond")
        return original_sha(path)

    resume.sha = growing_sha
    try:
        result = resume.atomic_copy_verified(src, dest)
    finally:
        resume.sha = original_sha
    assert result["status"] == "skipped_growing_log"
    assert not dest.exists()


def test_failure_logs_preserved(tmp_path):
    log = tmp_path / "run" / "fail.log"
    updates = []
    with pytest.raises(RuntimeError, match=str(log)):
        resume.run_stage(
            [sys.executable, "-c", "print('kept failure log'); raise SystemExit(7)"],
            log,
            30,
            lambda **kw: updates.append(kw),
            os.environ.copy(),
            tmp_path,
            sync=None,
        )
    assert "kept failure log" in log.read_text()
    assert updates and updates[0]["log"] == str(log)


def test_rng_restore_helper_cpu_smoke():
    src = Path("il/baselines/diffusion_policy/train_rgbd.py").read_text()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cpu_rng_state")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rng-helper", "exec"), scope)
    state = torch.get_rng_state().clone()
    mocked_gpu_like = state.clone().view(2, -1).t().contiguous().t()
    restored = scope["_cpu_rng_state"](mocked_gpu_like)
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    assert restored.is_contiguous()
    torch.set_rng_state(restored)


def test_no_forbidden_side_effect_tokens():
    text = Path(resume.__file__).read_text()
    assert "git push" not in text
    assert "submit" not in text.lower().replace("submission_executed", "")
    assert "dp_rgb_easy" not in text and "difficulty=hard" not in text
    assert "pip install" not in text and "uv pip" not in text
