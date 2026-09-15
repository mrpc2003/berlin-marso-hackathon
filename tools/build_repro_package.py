#!/usr/bin/env python3
"""Build a local, allowlisted float32 EMA release. Default is a full dry-run."""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validation_common import (
    ROOT, LEVELS, ENTRYPOINT, check_idle_gpu, checkpoint_contract, gpu_lock,
    load_manifest, new_output, require, safe_relative, seal_files, sha256, source_hashes,
    supervised, verify_seal, versions, write_json, require_gpu_lock,
)


def package_plan(manifest_path, *, allow_pending=True):
    return {"schema_version": 1, "status": "plan_only", "inputs": load_manifest(manifest_path, allow_pending=allow_pending),
            "input_manifest_sha256": sha256(manifest_path), "source_sha256": source_hashes(),
            "outputs": [f"checkpoints/rgb_dp_{l}.pt" for l in LEVELS] + ["submission.yaml", "MODEL_PROVENANCE.json", "release.tar.gz"],
            "weight_format": "float32 best EMA with exact tensor identity", "entrypoint": ENTRYPOINT,
            "artifact_policy": "only inputs.artifacts; no output-directory discovery",
            "gpu_smoke": "pending; run from a clean extraction with validated dependencies",
            "drive_destination": "marso/validations/<release_exp> (parent verified copy)"}


def assert_tensor_identity(source, compact):
    import torch
    require(set(source) == set(compact), "EMA state_dict keys changed")
    for key in source:
        a, b = source[key], compact[key]
        require(a.shape == b.shape and a.dtype == b.dtype, f"EMA shape/dtype changed: {key}")
        require(bool(torch.isfinite(b).all()), f"nonfinite compact tensor: {key}")
        # Byte equality also distinguishes +0 from -0. Flatten handles scalar buffers.
        require(torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)), f"EMA tensor changed: {key}")


def compact_checkpoint(entry, level, destination):
    import torch
    from tools.strip_ckpt import strip
    from warehouse_sort.il_policy import DEFAULT_DP_CONFIG, resolve_policy_config
    source = Path(entry["path"])
    require(source.is_file() and not source.is_symlink() and sha256(source) == entry["sha256"], f"{level}: source checkpoint SHA mismatch")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    cfg = checkpoint_contract(checkpoint, level, entry)
    ema = checkpoint.get("ema_agent") or checkpoint.get("model")
    overrides = {"act_horizon": entry["act_horizon"], "num_inference_steps": entry["num_inference_steps"], "seed": 0}
    strip(str(source), str(destination), fp16=False, config_overrides=overrides)
    compact = torch.load(destination, map_location="cpu", weights_only=False)
    assert_tensor_identity(ema, compact["model"])
    # Keep deployment and level provenance fields; omit arbitrary training args/absolute paths.
    resolved = resolve_policy_config(checkpoint, overrides)
    compact["config"] = {key: resolved[key] for key in DEFAULT_DP_CONFIG}
    for key in ("obs_mode", "state_dim", "act_dim", "image_hw", "max_episode_steps", "total_iters"):
        compact["config"][key] = cfg[key]
    compact["config"]["demo_paths"] = [f"demos/{level}/{Path(p).name}" for p in cfg["demo_paths"]]
    compact["source"] = entry["provenance_label"]
    torch.save(compact, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    assert_tensor_identity(ema, reloaded["model"])
    require(reloaded["config"] == compact["config"] and reloaded["source"] == entry["provenance_label"], "compact metadata roundtrip failed")
    require(sha256(source) == entry["sha256"], f"{level}: source changed while compacting")
    return {"source_sha256": entry["sha256"], "compact_sha256": sha256(destination),
            "source_label": entry["provenance_label"], "selection": entry["selection"],
            "source_weight_key": "ema_agent" if checkpoint.get("ema_agent") else "model",
            "iteration": checkpoint["iteration"], "training_completed_iterations": entry["training_completed_iterations"],
            "config_overrides": overrides, "loaded_policy_settings": reloaded["config"],
            "exact_tensor_identity": True, "finite": True, "floating_dtype": "float32"}


def archive_release(package, archive):
    """Sorted paths and fixed tar/gzip metadata make identical inputs reproducible."""
    with Path(archive).open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as tar:
                for path in sorted(Path(package).rglob("*")):
                    if not path.is_file():
                        continue
                    require(not path.is_symlink(), "symlink in release")
                    name = str(path.relative_to(package))
                    safe_relative(name)
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as f:
                        tar.addfile(info, f)


def build_worker(plan_path, out):
    plan = json.loads(Path(plan_path).read_text())
    require(source_hashes() == plan["source_sha256"], "source changed after plan")
    package = out / "release"
    package.mkdir(exist_ok=False)
    for name, digest in plan["source_sha256"].items():
        dst = package / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
        require(sha256(dst) == digest, f"source copy SHA mismatch: {name}")
    shutil.copyfile(ROOT / "docs/REPRODUCE.md", package / "README.md")
    shutil.copyfile(ROOT / "docs/REPRODUCE.md", package / "REPRODUCE.md")
    write_json(package / "SOURCE_MANIFEST.json", {"files": plan["source_sha256"],
               "capture": "allowlisted working tree bytes, including reviewed dirty overrides; not git HEAD export"})
    (package / "checkpoints").mkdir()
    provenance = {}
    sanitized_inputs = {"schema_version": 1, "original_manifest_sha256": plan["input_manifest_sha256"],
                        "checkpoints": {}, "artifacts": []}
    for level in LEVELS:
        entry = plan["inputs"]["checkpoints"][level]
        provenance[level] = compact_checkpoint(entry, level, package / "checkpoints" / f"rgb_dp_{level}.pt")
        sanitized_inputs["checkpoints"][level] = {k: v for k, v in entry.items() if k != "path"}
    write_json(package / "MODEL_PROVENANCE.json", provenance)
    submission = {"team": "mrpc2003", "rgb": {"policy": ENTRYPOINT,
                  "levels": {l: {"checkpoint": f"checkpoints/rgb_dp_{l}.pt"} for l in LEVELS}}}
    # JSON is valid YAML and avoids an extra serializer dependency in this command.
    write_json(package / "submission.yaml", submission)
    for artifact in plan["inputs"].get("artifacts", []):
        path = Path(artifact["path"])
        require(path.is_file() and not path.is_symlink() and sha256(path) == artifact["sha256"], "explicit artifact SHA mismatch")
        dst = package / "artifacts" / safe_relative(artifact["name"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        require(sha256(dst) == artifact["sha256"], "artifact copy failed")
        sanitized_inputs["artifacts"].append({"name": f"artifacts/{artifact['name']}", "sha256": artifact["sha256"]})
    write_json(package / "PACKAGE_INPUT.json", sanitized_inputs)
    environment = versions()
    missing = [n for n, d in environment["packages"].items() if d["version"] is None]
    require(not missing, f"validated runtime dependencies missing: {missing}; no automatic install")
    write_json(package / "PACKAGE_VERSIONS.json", environment)
    (package / "requirements-repro.txt").write_text("# Exact installed versions captured at build; consult pixi.toml for declared constraints.\n" +
        "\n".join(f"{name}=={data['version']}" for name, data in environment["packages"].items()) + "\n")
    write_json(package / "THIRD_PARTY_LICENSES.json", environment["packages"])
    (package / "LICENSES.md").write_text(
        "# License provenance\n\nExisting allowlisted LICENSE files are preserved. "
        "This repository snapshot has no additional license grant from the packager. "
        "If no project LICENSE is present, redistribution rights must be resolved by the parent before sharing. "
        "THIRD_PARTY_LICENSES.json records installed dependency license metadata; dependency code is not bundled. "
        "Installers must retain the license files supplied by those distributions.\n")
    require(source_hashes() == plan["source_sha256"], "source changed during packaging")
    seal_files(package)
    count = verify_seal(package)
    archive = out / "release.tar.gz"
    archive_release(package, archive)
    # Verify actual archive bytes by extraction, without importing original checkout modules.
    clean = out / "archive_verify"
    clean.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            safe_relative(member.name)
            require(member.isfile(), "non-file member in release")
        tar.extractall(clean, filter="data")
    require(verify_seal(clean) == count, "archive roundtrip failed")
    # Keep one compact release and archive; this temporary verified extraction is regenerable.
    shutil.rmtree(clean)
    write_json(out / "build_result.json", {"status": "built_local", "archive": "release.tar.gz", "archive_sha256": sha256(archive),
               "verified_files": count, "clean_extract_sha": "passed", "clean_extract_gpu_smoke": "pending", "drive_copy": "pending_parent_verified_copy"})


def smoke_worker(root, out):
    """One tiny reset/action/step episode per level; no performance claim."""
    require_gpu_lock()
    import torch
    from tools.run_generalization import CheckedPolicy, bind_actual_budget, frozen_protocols, resolved_config
    from warehouse_sort.utils import make_env, to_device
    require(ROOT.resolve() == root.resolve(), "smoke must execute the script inside the clean extraction")
    require(not (root / ".git").exists(), "smoke requires a clean extracted folder")
    require(torch.cuda.is_available(), "GPU smoke requires CUDA")
    n_files = verify_seal(root)
    submission = json.loads((root / "submission.yaml").read_text())
    require(submission["rgb"]["policy"] == ENTRYPOINT and set(submission["rgb"]["levels"]) == set(LEVELS), "submission contract mismatch")
    module, fn = submission["rgb"]["policy"].split(":")
    mod = importlib.import_module(module)
    require(Path(mod.__file__).resolve().is_relative_to(root), "policy imported from outside extraction")
    provenance = json.loads((root / "MODEL_PROVENANCE.json").read_text())
    results = {}
    for level in LEVELS:
        path = root / safe_relative(submission["rgb"]["levels"][level]["checkpoint"])
        require(sha256(path) == provenance[level]["compact_sha256"], "compact SHA differs from provenance")
        spec = dict(frozen_protocols()["protocols"]["fresh_seed"][level], num_envs=1)
        cfg = resolved_config(level, spec, str(path))
        env, _ = make_env(cfg, "rgb", cfg.randomization, num_envs=1)
        try:
            actual_budget = bind_actual_budget(env, spec)
            obs, _ = env.reset(seed=[9100])
            # Exercise the submission entrypoint with exactly its four required arguments.
            agent = getattr(mod, fn)(str(path), to_device(obs, torch.device("cuda")), env.single_action_space, torch.device("cuda"))
            for key, value in provenance[level]["config_overrides"].items():
                require(agent.cfg[key] == value, f"embedded deployment setting changed: {key}")
            action = CheckedPolicy(agent).act(to_device(obs, torch.device("cuda")))
            env.step(action)
            results[level] = {"episodes": 1, "step_calls": 1, "action_shape": list(action.shape), "action_finite": True,
                              "num_parcels": env.unwrapped.num_parcels, "max_episode_steps": int(env.unwrapped.max_episode_steps),
                              "actual_budget": actual_budget,
                              "loaded_policy_settings": agent.cfg, "checkpoint_sha256": sha256(path)}
        finally:
            env.close()
    write_json(out / "smoke_result.json", {"status": "passed", "verified_files": n_files, "levels": results,
               "scope": "same validated environment, clean extracted source, tiny one-step episodes; not fresh-machine installation or performance evaluation",
               "versions": versions()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "docs/VALIDATION_INPUT.example.json")
    ap.add_argument("--out", type=Path, default=Path("/content/marso-validation"))
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke", action="store_true", help="verify clean extraction and run tiny GPU episodes")
    modes.add_argument("--verify-only", action="store_true", help="check extracted archive file hashes without runtime")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--worker", nargs=3, metavar=("MODE", "PLAN_OR_ROOT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        mode, target, out = args.worker
        if mode == "build":
            build_worker(target, Path(out))
        elif mode == "smoke":
            smoke_worker(Path(target), Path(out))
        else:
            raise ValueError("unknown worker")
        return
    if args.verify_only:
        print(json.dumps({"sha_verified_files": verify_seal(ROOT)}))
        return
    require(0 < args.timeout <= 43200, "invalid timeout")
    if args.smoke:
        require(not Path(args.out).resolve().is_relative_to(ROOT), "smoke output must be outside clean extraction")
        with gpu_lock() as lock_fd:
            idle = check_idle_gpu()
            out = new_output(args.out, "smoke_")
            write_json(out / "preflight.json", idle)
            try:
                supervised([sys.executable, str(Path(__file__).resolve()), "--worker", "smoke", str(ROOT), str(out)],
                           out, args.timeout, cwd=ROOT, lock_fd=lock_fd)
            finally:
                seal_files(out)
                print(f"smoke records: {out}")
        return
    plan = package_plan(args.manifest, allow_pending=not args.run)
    plan["execution"] = {"python": sys.executable, "repo": str(ROOT), "out_parent": str(args.out.resolve()),
                         "timeout_seconds": args.timeout, "worker_command_template": [sys.executable,
                             str(Path(__file__).resolve()), "--worker", "build", "<new-output>/plan.json", "<new-output>"]}
    if not args.run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    out = new_output(args.out, "release_")
    write_json(out / "plan.json", plan)
    try:
        supervised([sys.executable, str(Path(__file__).resolve()), "--worker", "build", str(out / "plan.json"), str(out)], out, args.timeout)
    finally:
        seal_files(out)
        print(f"local package records: {out}")


if __name__ == "__main__":
    main()
