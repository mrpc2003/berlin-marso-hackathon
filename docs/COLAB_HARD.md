# Hard RGB Diffusion Policy on the existing Colab runtime

Worktree-only implementation. No training, environment installation, data transfer, or submission
is performed by generating this notebook. Generate from the reviewed worktree with:

```sh
python3 -m tools.make_colab_hard_notebook
```

Open `MARSO_COLAB_HARD.ipynb` in the existing runtime. **Do not use Run All.** Execute each stage
explicitly: mount check → source setup → checks → smoke → full launch → monitor. Setup, checks,
smoke, full launch (`RUN_HARD=False`), and optional final sync all default off.

The runtime must already contain `/content/berlin-marso-hackathon` and the validated
`/content/marso-py312/bin/python`. Setup accepts only embedded base/current source hashes,
backs up replaced originals, snapshots the complete required Python/YAML sources, and rejects
unknown patches, symlinks, escapes, and existing output directories (including empty ones).
It never clones, reinstalls, restarts the kernel, or reads credentials. Parent stages these files:

```text
/content/berlin-marso-hackathon/il/demos/hard/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5
/content/berlin-marso-hackathon/il/demos/hard/trajectory.rgb.pd_ee_delta_pos.physx_cuda.json
```

Drive must be a real mount at `/content/drive`, with existing `MyDrive/marso`. A Hard demo marker
on Drive is not required. Dataset checks read all 200 trajectories, RGB frames, finite actions
and 26-dimensional proprioception, six-parcel metadata and Hard randomization, and hash both
input files before/after validation. Metadata episode-count discrepancies are recorded honestly.
The reference checks actual GPU reset/step, scene RGB, six parcels, the 800-step limit, observed
pose/bin variation over 16 fixed resets, rendering, and one scripted episode. A legitimate
scripted sort failure is reported without rejecting an otherwise valid environment.
The active limit is read with `env.get_wrapper_attr('_max_episode_steps')` and must be 800;
missing or different wrapper limits fail even when both config limits say 800. The reference
report's `max_episode_steps` uses that verified wrapper value. It separately records
`wrapper_max_episode_steps`, `configured_max_episode_steps`, `difficulty_max_episode_steps`,
`base_max_episode_steps`, and `spec_max_episode_steps`. On the observed T4 wrapper chain the
active limit is 800, spec metadata is None, and the base constructor metadata is 100.
This check does not change any environment field or rollout behavior.

Smoke uses all 200 demos, batch 128 and exactly 100 processed batches on CUDA, with separate checkpoints.
The Hard worker observes all 100 finite losses and real AdamW calls. Successful updates must be
positive, have finite unscaled gradients and resulting state, and agree with the checkpoint's
optimizer counters. Normal initial GradScaler overflow recovery is allowed (the CPU regression
processes 100 batches with 98 successful updates and 2 skips). `smoke/batch-audit.json` and
`run/smoke.json` record processed, actual, and skipped counts. Scheduler and EMA still advance
once per batch, including skipped AMP updates; both must reach 100. Full training
starts from seed 1 with no resume input: 40,000 iterations, lr 1e-4, ResNet18 scene RGB,
horizons 2/8/16, AMP, checkpoint every 1,000, eval every 5,000 with 32 episodes / 8 envs.
The fixed seed does not promise bitwise determinism; the existing preset retains cuDNN benchmarking.

Every GPU stage first takes the machine-wide `/tmp/marso-active-gpu.lock`, shared with the
package/generalization worker, then a repository lock and experiment lock. The actual GPU process
verifies all three inherited FDs against their expected inodes, confirms an independent open is
blocked, and confirms the inherited description owns the lock before importing CUDA-capable code.
Hard uses `tools/hard_gpu_worker.py` to compose the same Hydra method/overrides and reuse the
baseline flag conversion, then executes `train_rgbd.py` in that same process. Baseline `il/train.py`
is preserved. Stage logs retain the dispatcher command header; resolved Hydra config and launcher
command/PID evidence are saved under `<smoke|train>/hydra/` (Hydra job log: `hard_gpu_worker.log`).
Eval, diagnostics and probes also execute inside the verified worker. The worker retains the FDs
even if the supervisor dies. No GPU-related process is launched during local CPU verification.

Training has a six-hour deadline and 60-second progress reports. Cleanup targets only the process
group created for the stage. It waits for group liveness independently of the leader, escalates to
SIGKILL for surviving descendants, and fails if cleanup cannot be verified. A successful leader
exit with surviving descendants also fails the stage and triggers cleanup. Linux zombie entries
are distinguished from live processes. CPU integration tests use real descendants and signals;
the independent duplicate-GPU `ps` gate is bypassed only in fixtures because the local sandbox
blocks `ps`. Production still checks that gate before launch.

New artifacts live under `/content/marso-hard/hard_YYYYMMDD_HHMMSS`; a bounded sync worker mirrors
them to `MyDrive/marso/recoveries/hard/<same-name>`. Periodic sync has a 45-second timeout and is
nonfatal. Copies use temporary files and rehash the source and destination; mount loss prevents
promotion. A final verification worker has a 600-second timeout. Logs, source snapshots/backups,
plan, input provenance/schema, checks, smoke, checkpoints, progress, evaluation JSONL, video,
diagnostics/traces, and summary stay local first.

Posttraining validates final iteration/scheduler 40,000 and finite agent/EMA with durable
optimizer/scheduler/EMA/scaler/RNG state. Legacy `args.num_parcels` is not used as scene evidence.
The training-time best must be from the exact first iteration attaining the final history's maximum,
and its entire evaluation history must equal the corresponding final-history prefix. Later ties
are rejected, matching the trainer's strict-improvement rule. Final is a recorded fallback only if no
best file was produced and all observed sort metrics were zero. Four 32-episode seed-5000
sweeps use `(act_horizon, diffusion_steps)` = `(8,16), (4,16), (8,32), (4,32)`. The highest sort
accuracy wins (ties use listed order), followed by a one-env video and eight diagnostic episodes.
Scores are **local validation**, never official/heldout. Existing eval/diagnostic loops execute
799 steps with the configured 800-step limit; shared baseline code is preserved.
All numeric eval fields must be finite and satisfy metric ranges. Diagnostics require eight seeds,
six parcels, finite JSON and all six NPZ arrays with exact shapes/dtypes and valid gripper/count
ranges. JSON phase/error summaries are recomputed from traces, including the logger's unused
800th padding row. These validations run before compute completion and again before any durability
receipt, so matching hashes cannot certify corrupt diagnostic numbers.

`run/status.json` separates `compute_status` from `remote_status`. Only successful verification
of every immutable payload and required artifact produces `run/durability.json` and a completed
status. Status and the receipt itself are excluded from the receipt manifest. If compute finishes
with Drive pending, remount and use the notebook's optional final-sync cell, or:

```sh
/content/marso-py312/bin/python /content/berlin-marso-hackathon/tools/run_colab_hard.py \
  /content/marso-hard/hard_<timestamp>/run/plan.json --phase finalize
```

Restore the exact saved `PLAN` path after reconnect. Never rerun setup or a training phase against
an existing experiment. This pipeline intentionally does not auto-resume after a lost runtime;
local/Drive `latest.pt` preserves recovery state for a separately reviewed continuation. Gates
that fail leave evidence and require a new experiment, not reuse of partial outputs.

Local tests use the parent's existing CPU environment without installing anything:

```sh
PYTHONDONTWRITEBYTECODE=1 ../../.venv-cpu/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_colab_hard.py tests/test_hard_processes.py tests/test_hard_notebook.py
```

CPU tests and notebook AST/nbformat validation do not establish CUDA training or simulator results.
Those gates remain for parent execution in the existing GPU runtime.

`SOURCE_FILES` is the exact required and allowed reviewed payload set (41 paths). Bootstrap,
plan loading, source checks and notebook generation reject omitted or unexpected entries without
depending on a remote Git inventory. Local Git is used only while generating base hashes.
`tools/audit_hard_notebook.py` validates the saved notebook against the current generator and every
embedded source, parses all Python, requires nbformat validation, checks unexecuted cells and False
switches, and scans decoded sources/cells/metadata plus Hard tests/docs. Scan results contain only
path, line and rule; never secret contents. nbformat must already be available; no installer is run.

Parent reports that Hard H5 and JSON are already uploaded and SHA-verified on the existing Colab.
This fix pass does not repeat those actions or execute any notebook cell. Independent approval is
still pending. Outstanding runtime gates: target Python 3.12/T4 dependencies, machine-lock interaction
with the actual package worker, CUDA/reference, smoke, fresh 40,000-batch training, evaluation,
video/diagnostics, and live Drive durability. CPU torch is 2.14.0; target torch is 2.11.0.
