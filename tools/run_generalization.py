#!/usr/bin/env python3
"""Frozen local evaluation. Default: full dry-run. GPU work requires --run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validation_common import (
    ROOT, LEVELS, WEIGHTS, ENTRYPOINT, GPU_LOCK, append_jsonl, checkpoint_contract,
    check_idle_gpu, gpu_lock, load_manifest, new_output, require, seal_files, sha256,
    source_hashes, supervised, versions, write_json, require_gpu_lock,
)


def frozen_protocols():
    return json.loads((ROOT / "docs/VALIDATION_PROTOCOLS.json").read_text())


def make_plan(manifest_path, *, allow_pending=True):
    manifest = load_manifest(manifest_path, allow_pending=allow_pending)
    return {"schema_version": 1, "status": "plan_only", "input_manifest_sha256": sha256(manifest_path),
            "protocol_sha256": sha256(ROOT / "docs/VALIDATION_PROTOCOLS.json"),
            "frozen": frozen_protocols(), "inputs": manifest, "source_sha256": source_hashes(),
            "entrypoint": ENTRYPOINT, "stages": [f"{p}/{l}" for p in ("fresh_seed", "stress") for l in LEVELS],
            "total_scored_episodes": 600, "extra_policy_rollouts_for_logging": 0,
            "drive_destination": "marso/validations/<validation_exp> (parent verified copy)",
            "gpu_proof": "pending", "progress_seconds": 30, "default_stage_timeout_seconds": 7200}


def validate_episode(row, spec):
    require(type(row.get("seed")) is int and row["seed"] in spec["seeds"], "unexpected episode seed")
    require(row["num_parcels"] == spec["num_parcels"] and row["max_episode_steps"] == spec["max_episode_steps"], "episode geometry/budget mismatch")
    require(row["step_calls"] == spec["rollout_step_calls"], "unexpected rollout step count")
    for key in ("sorted", "mis_sorted"):
        x = row[key]
        require(type(x) in (float, int) and math.isfinite(x) and x == int(x) and 0 <= x <= spec["num_parcels"], f"invalid {key}")
    require(type(row["all_placed"]) is bool, "invalid all_placed")
    x = row["steps_to_complete"]
    require(type(x) in (int, float) and math.isfinite(x) and x == int(x) and 0 <= x <= spec["max_episode_steps"], "invalid steps_to_complete")
    # Sticky mis-sort history can overlap later correct placement; do not sum the two.


def aggregate_rows(rows, spec):
    require(len(rows) == spec["n_episodes"], "incomplete episode count")
    require([r["seed"] for r in rows] == spec["seeds"], "missing, duplicate, or reordered seeds")
    for row in rows:
        validate_episode(row, spec)
    n, p = len(rows), spec["num_parcels"]
    return {"n_episodes": n, "num_parcels": p,
            "sort_accuracy": sum(r["sorted"] for r in rows) / (n * p),
            "mean_sorted": sum(r["sorted"] for r in rows) / n,
            "mis_sort_rate": sum(r["mis_sorted"] for r in rows) / (n * p),
            "all_placed_rate": sum(r["all_placed"] for r in rows) / n,
            "mean_steps": sum(r["steps_to_complete"] for r in rows) / n}


def validate_metrics(metrics, rows, spec):
    expected = aggregate_rows(rows, spec)
    for key, value in expected.items():
        actual = metrics[key]
        require(type(actual) in (int, float) and math.isfinite(actual) and abs(actual - value) < 1e-7,
                f"aggregate disagrees with episode rows: {key}")
    return expected


def weighted_score(by_level):
    require(set(by_level) == set(LEVELS), "weighted result requires all three levels; no renormalization")
    for level in LEVELS:
        x = by_level[level]["sort_accuracy"]
        require(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1, "invalid level accuracy")
    return sum(WEIGHTS[l] * by_level[l]["sort_accuracy"] for l in LEVELS)


def resolved_config(level, spec, checkpoint):
    from omegaconf import OmegaConf
    from warehouse_sort.utils import compose_cfg
    cfg = compose_cfg([f"difficulty={level}"], config_dir=str(ROOT / "conf"))
    require(int(cfg.difficulty.num_parcels) == spec["num_parcels"] and int(cfg.max_episode_steps) == spec["max_episode_steps"], "difficulty baseline changed since protocol freeze")
    cfg.difficulty.fixed_poses = spec["fixed_poses"]
    cfg.randomization = OmegaConf.create(spec["randomization"])
    for key in ("num_envs", "obs_mode", "obs_camera", "control_mode", "camera"):
        cfg[key] = spec[key]
    cfg.device = "cuda"
    cfg.checkpoint = checkpoint
    cfg.policy = ENTRYPOINT
    cfg.record_video = False
    return cfg


def bind_actual_budget(env, spec):
    """Gym consumes max_episode_steps for its wrapper, not the task constructor.

    Align WarehouseSort's completion sentinel before the first measured reset;
    do not change the wrapper budget or the shared rollout loop.
    """
    wrapped = getattr(env, "_env", env)
    wrapper_steps = int(wrapped.get_wrapper_attr("_max_episode_steps"))
    require(wrapper_steps == spec["max_episode_steps"], "actual TimeLimit budget mismatch")
    base = env.unwrapped
    previous = int(base.max_episode_steps)
    base.max_episode_steps = wrapper_steps
    return {"wrapper_max_episode_steps": wrapper_steps, "task_budget_before_binding": previous,
            "task_max_episode_steps": int(base.max_episode_steps)}


class CheckedPolicy:
    """Observe the policy contract without access to privileged diagnostics."""
    def __init__(self, agent):
        self.agent = agent
        self.calls = 0

    def reset(self):
        self.agent.reset()

    def act(self, obs, deterministic=True):
        import torch
        require(set(obs) == {"rgb", "state"} and obs["state"].shape[1] == 26, "policy observation must be RGB + proprio only")
        a = self.agent.act(obs, deterministic=deterministic)
        require(a.shape == (obs["state"].shape[0], 4), "expected finite Nx4 action")
        require(bool(torch.isfinite(a).all()) and bool((a.abs() <= 1).all()), "nonfinite or out-of-range action")
        self.calls += 1
        return a


def pose_record(base, slot, seed):
    """Actual post-reset poses. Read-only, separate from policy observations."""
    return {"seed": int(seed), "fixed_poses": bool(base.fixed_poses),
            "parcel_poses": [p.pose.raw_pose[slot].detach().cpu().tolist() for p in base.parcels],
            "bin_poses": [b.pose.raw_pose[slot].detach().cpu().tolist() for b in base.bins],
            "nominal": {"inbound_center": list(base.inbound_center), "parcel_half": list(base.parcel_half),
                        "bin_base_x": float(base.bin_base_x), "bin_base_y": float(base.bin_base_y)},
            "diagnostic_only": True}


def pose_deltas(row, spec):
    require(row["fixed_poses"] is spec["fixed_poses"], "actual fixed_poses override ignored")
    p = spec["num_parcels"]
    require(len(row["parcel_poses"]) == p and len(row["bin_poses"]) == 2, "pose count mismatch")
    rand, nominal = spec["randomization"], row["nominal"]
    xy, yaw, binxy = [], [], []
    for j, pose in enumerate(row["parcel_poses"]):
        require(len(pose) == 7 and all(math.isfinite(x) for x in pose), "invalid parcel pose")
        r, c = divmod(j, 2)
        gx = nominal["inbound_center"][0] + (c - .5) * (2 * nominal["parcel_half"][0] + .06)
        gy = nominal["inbound_center"][1] + (r - (math.ceil(p / 2) - 1) / 2) * (2 * nominal["parcel_half"][1] + .06)
        xy.extend([pose[0] - gx, pose[1] - gy])
        yaw.append(2 * math.atan2(pose[6], pose[3]))
    for j, pose in enumerate(row["bin_poses"]):
        require(len(pose) == 7 and all(math.isfinite(x) for x in pose), "invalid bin pose")
        binxy.extend([pose[0] - nominal["bin_base_x"], abs(pose[1]) - nominal["bin_base_y"]])
    swapped = row["bin_poses"][0][1] < 0
    require((row["bin_poses"][0][1] * row["bin_poses"][1][1]) < 0, "bins do not occupy opposite sides")
    require(rand["bin_position"]["side_swap_prob"] != 0 or not swapped, "unexpected bin side swap")
    for values, bounds, name in ((xy, rand["parcel_pose"]["xy_jitter"], "parcel xy"),
                                  (yaw, rand["parcel_pose"]["yaw_jitter"], "yaw"),
                                  (binxy, rand["bin_position"]["xy_jitter"], "bin xy")):
        require(all(bounds[0] - 1e-5 <= x <= bounds[1] + 1e-5 for x in values), f"actual {name} outside frozen range")
    return xy, yaw, binxy, swapped


def verify_pose_effect(rows, spec):
    require([r["seed"] for r in rows] == spec["seeds"], "incomplete reset diagnostics")
    deltas = [pose_deltas(r, spec) for r in rows]
    for col, bounds in ((0, spec["randomization"]["parcel_pose"]["xy_jitter"]),
                        (1, spec["randomization"]["parcel_pose"]["yaw_jitter"]),
                        (2, spec["randomization"]["bin_position"]["xy_jitter"])):
        # Check variation across seeds for the same coordinate, not only across parcels.
        values = [d[col][0] for d in deltas]
        if bounds[1] > bounds[0]:
            require(max(values) - min(values) > 1e-6, "randomization had no observed effect across seeds")
    if spec["randomization"]["bin_position"]["side_swap_prob"] == .5:
        require({d[3] for d in deltas} == {False, True}, "bin swapping had no observed effect")
    return {"status": "verified_actual_reset_poses", "n_resets": len(rows), "diagnostic_only": True}


def write_rgb(path, rgb):
    import numpy as np
    data = rgb.detach().cpu().numpy()
    require(data.ndim == 3 and data.shape[-1] == 3 and data.dtype == np.uint8, "unexpected reset RGB render")
    h, w, _ = data.shape
    Path(path).write_bytes(f"P6\n{w} {h}\n255\n".encode() + data.tobytes())


def run_worker(plan_path, protocol, level, out):
    require_gpu_lock()
    import torch
    from omegaconf import OmegaConf
    from warehouse_sort.utils import load_agent, make_env, rollout_metrics
    require(torch.cuda.is_available(), "CUDA GPU required; no silent CPU fallback")
    plan = json.loads(Path(plan_path).read_text())
    require(source_hashes() == plan["source_sha256"], "source changed after plan freeze")
    spec = plan["frozen"]["protocols"][protocol][level]
    entry = plan["inputs"]["checkpoints"][level]
    require(sha256(entry["path"]) == entry["sha256"], "checkpoint SHA changed")
    ckpt = torch.load(entry["path"], map_location="cpu", weights_only=False)
    checkpoint_contract(ckpt, level, entry)
    del ckpt
    cfg = resolved_config(level, spec, entry["path"])
    cfg.policy_kwargs = {"act_horizon": entry["act_horizon"], "num_inference_steps": entry["num_inference_steps"], "seed": 0}
    write_json(out / "resolved_eval_config.json", {"cfg": OmegaConf.to_container(cfg, resolve=True), "eval": spec})
    env, _ = make_env(cfg, "rgb", cfg.randomization, num_envs=spec["num_envs"])
    rows, diagnostics = [], []
    start = time.monotonic()
    try:
        base = env.unwrapped
        write_json(out / "actual_budget.json", bind_actual_budget(env, spec))
        require(base.num_parcels == spec["num_parcels"] and base.max_episode_steps == spec["max_episode_steps"], "actual environment budget/count mismatch")
        require(bool(base.fixed_poses) == spec["fixed_poses"], "actual fixed_poses override ignored")
        agent, _ = load_agent(entry["path"], env, torch.device("cuda"), entrypoint=ENTRYPOINT,
                              policy_kwargs=OmegaConf.to_container(cfg.policy_kwargs, resolve=True))
        require(sha256(entry["path"]) == entry["sha256"], "checkpoint changed while loading policy")
        for key in ("act_horizon", "num_inference_steps", "seed"):
            require(agent.cfg[key] == cfg.policy_kwargs[key], f"loaded policy ignored {key}")
        require(agent.act_horizon == entry["act_horizon"] and
                len(agent.agent.noise_scheduler.timesteps) == entry["num_inference_steps"] and
                agent.generator.initial_seed() == 0, "actual loaded inference settings mismatch")
        write_json(out / "loaded_policy.json", {"checkpoint_sha256": entry["sha256"], "settings": agent.cfg,
                   "versions": versions(), "actual_randomization": base._rand, "device": str(agent.device),
                   "gpu_name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                   "actual_environment": {"num_envs": int(base.num_envs), "num_parcels": int(base.num_parcels),
                       "max_episode_steps": int(base.max_episode_steps), "fixed_poses": bool(base.fixed_poses),
                       "camera_width": int(base.camera_width), "camera_height": int(base.camera_height),
                       "robot_init_qpos_noise": float(base.robot_init_qpos_noise)}})
        checked = CheckedPolicy(agent)

        def on_reset(actual, obs, seeds):
            for i, seed in enumerate(seeds):
                row = pose_record(actual, i, seed)
                pose_deltas(row, spec)
                if len(diagnostics) < 2:
                    filename = f"reset_seed_{seed}.ppm"
                    write_rgb(out / filename, obs["rgb"][i])
                    row["render"] = filename
                    row["render_sha256"] = sha256(out / filename)
                diagnostics.append(row)
                append_jsonl(out / "diagnostics.jsonl", row)

        def on_batch(actual, ev, seeds):
            for i, seed in enumerate(seeds):
                row = {"seed": int(seed), "num_parcels": int(actual.num_parcels), "max_episode_steps": int(actual.max_episode_steps),
                       "step_calls": int(actual.elapsed_steps[i].item()), "sorted": float(ev["success_count"][i].item()),
                       "mis_sorted": float(ev["mis_sort_count"][i].item()), "all_placed": bool(ev["all_placed"][i].item()),
                       "steps_to_complete": float(ev["steps_to_complete"][i].item())}
                validate_episode(row, spec)
                rows.append(row)
                append_jsonl(out / "episodes.jsonl", row)
            print(f"[{protocol}/{level}] {len(rows)}/{spec['n_episodes']} episodes", flush=True)

        metrics = rollout_metrics(env, checked, torch.device("cuda"), spec["n_episodes"], spec["seeds"],
                                  spec["max_episode_steps"], on_reset=on_reset, on_batch=on_batch)
        validate_metrics(metrics, rows, spec)
        require(checked.calls == (spec["n_episodes"] // spec["num_envs"]) * spec["rollout_step_calls"], "unexpected number of action batches")
        effect = verify_pose_effect(diagnostics, spec)
        result = {"protocol": protocol, "level": level, "metrics": metrics, "pose_effect": effect,
                  "checkpoint_sha256": entry["sha256"], "max_episode_steps": spec["max_episode_steps"],
                  "step_calls_per_episode": spec["rollout_step_calls"], "elapsed_seconds": time.monotonic() - start}
        append_jsonl(out / "metrics.jsonl", result)
        write_json(out / "result.json", result)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "docs/VALIDATION_INPUT.example.json")
    ap.add_argument("--out", type=Path, default=Path("/content/marso-validation"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-timeout", type=int, default=7200)
    ap.add_argument("--worker", nargs=4, metavar=("PLAN", "PROTOCOL", "LEVEL", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        plan, protocol, level, out = args.worker
        run_worker(plan, protocol, level, Path(out))
        return
    plan = make_plan(args.manifest, allow_pending=not args.run)
    require(0 < args.stage_timeout <= 43200, "invalid stage timeout")
    plan["execution"] = {"python": sys.executable, "repo": str(ROOT), "out_parent": str(args.out.resolve()),
                         "stage_timeout_seconds": args.stage_timeout, "gpu_lock": str(GPU_LOCK),
                         "worker_command_template": [sys.executable, str(Path(__file__).resolve()), "--worker",
                             "<new-output>/plan.json", "<protocol>", "<level>", "<new-output>/<protocol>_<level>"]}
    if not args.run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    with gpu_lock() as lock_fd:
        idle = check_idle_gpu()
        # Hash/load validation runs inside bounded workers; manifest structure is already frozen.
        out = new_output(args.out, "validation_")
        write_json(out / "plan.json", plan)
        write_json(out / "preflight.json", idle)
        try:
            results = {}
            for protocol in ("fresh_seed", "stress"):
                results[protocol] = {}
                for level in LEVELS:
                    stage = out / f"{protocol}_{level}"
                    stage.mkdir(exist_ok=False)
                    supervised([sys.executable, str(Path(__file__).resolve()), "--worker", str(out / "plan.json"), protocol, level, str(stage)],
                               stage, args.stage_timeout, lock_fd=lock_fd)
                    result = json.loads((stage / "result.json").read_text())
                    require(result["protocol"] == protocol and result["level"] == level and
                            result["checkpoint_sha256"] == plan["inputs"]["checkpoints"][level]["sha256"],
                            "stage identity does not match frozen checkpoint mapping")
                    rows = [json.loads(line) for line in (stage / "episodes.jsonl").read_text().splitlines()]
                    validate_metrics(result["metrics"], rows, plan["frozen"]["protocols"][protocol][level])
                    results[protocol][level] = result["metrics"]
            write_json(out / "summary.json", {"status": "completed", "official_score": False, "results": results,
                       "weighted_proxy_scores": {p: weighted_score(r) for p, r in results.items()}, "weights": WEIGHTS,
                       "confidence_intervals": "not computed; parcels are not independent trials", "drive_copy": "pending_parent_verified_copy"})
        except BaseException as e:
            write_json(out / "failure.json", {"status": "failed", "error": str(e)})
            raise
        finally:
            seal_files(out)
            print(f"local records: {out}", flush=True)


if __name__ == "__main__":
    main()
