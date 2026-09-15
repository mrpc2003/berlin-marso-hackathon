# Frozen local validation protocol v1

Frozen before any new results. The machine-readable authority is
`docs/VALIDATION_PROTOCOLS.json`: it contains all 100 seeds for each level and
protocol. Its SHA256, every allowlisted source hash, the input manifest hash, and
full resolved configurations are saved before evaluation. The Easy/Medium input
JSON is evidence data, never executable instructions. No Hard result is assumed.

| Protocol | Episodes per level | Seeds | Meaning |
|---|---:|---|---|
| fresh_seed | 100 | 6000–6099 | New seeds within original difficulty ranges. Easy is **repeatability on a fixed layout**, not spatial generalization. |
| stress | 100 | 8000–8099 | Our own perturbation proxy, **not the official hidden judge**. |

These seeds are disjoint from checkpoint selection seeds 5000+ and internal
training evaluations starting at 0. New outcomes must not select another model,
change deployment knobs, or revise the frozen protocol without a new version.

| Protocol / level | Parcel XY (m) | Yaw (rad) | Bin swap probability | Bin XY (m) | fixed_poses | Parcels / max steps |
|---|---|---|---:|---|---|---|
| fresh_seed / Easy | 0 | 0 | 0 | 0 | true | 2 / 250 |
| fresh_seed / Medium | ±0.015 | 0 | 0 | 0 | false | 4 / 500 |
| fresh_seed / Hard | ±0.020 | ±0.10 | 0.5 | 0 | false | 6 / 800 |
| stress / Easy | ±0.010 | ±0.05 | 0 | ±0.005 | **false** | 2 / 250 |
| stress / Medium | ±0.025 | ±0.05 | 0 | ±0.005 | false | 4 / 500 |
| stress / Hard | ±0.030 | ±0.15 | 0.5 | ±0.010 | false | 6 / 800 |

Four environments run sequential batches of four (25 batches per stage), six
stages total. This is frozen because batch size changes the random-number stream.
The existing `warehouse_sort.utils.rollout_metrics` used by `eval.py` executes
`max_steps - 1` steps: 249 / 499 / 799. We preserve that algorithm and record both
the configured budget and actual calls. Changing it would change comparability
with selection results. Sticky placement/mis-sort semantics are unchanged;
mis-sort history can overlap later correct placements.

Gym consumes `max_episode_steps` for the TimeLimit wrapper, while the task
constructor has its own completion-time sentinel. Before the first measured
reset, this runner verifies the actual wrapper limit and explicitly binds the
task sentinel to the same 250 / 500 / 800 budget. It records both previous and
effective values in `actual_budget.json`; episode `step_calls` comes from the
actual environment elapsed counter. This leaves the rollout algorithm and
sticky sort counts unchanged while fixing the completion-time sentinel within
this evaluation lane. See the [upstream TimeLimit implementation](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/registration.py).

`tools/rollout_logger.py` currently handles only one vector batch and allocates
per-step trace arrays. It is preserved byte for byte. The only existing-source
change is two optional read-only observers in `warehouse_sort/utils.py`:
`on_reset(base, obs, seeds)` and `on_batch(base, final_evaluation, seeds)`.
Default callers behave identically. Observers record all 100 outcomes from the
same rollout; there is no second 100-episode logging pass.

The policy receives scene RGB (128×128) and the existing 26 proprioceptive
features only. The inference noise generator starts at seed 0 once per level /
protocol and advances normally through its batches, matching the loader's
semantics; it is not re-seeded each episode. Winner action horizon and diffusion
steps come from the explicit checkpoint mapping. Loaded settings are checked and
recorded. The actual action must be finite, shape N×4, and within [-1, 1].

Reset diagnostics read actual parcel and bin poses, outside the policy path.
They verify counts, the actual `fixed_poses` flag, bounds and across-seed jitter
(including yaw, bin jitter and both Hard swap states). Easy stress must change
its actual layout. Each stage saves the first two real reset RGB frames as PPM
images plus SHA256 references, and 100 small pose rows. A diagnostic failure
invalidates that stage; it is never silently ignored. These are reset images,
not rollout videos or a simulator-performance proof made on this Mac.

Each stage writes `resolved_eval_config.json`, `loaded_policy.json` (including
package versions and normalized actual randomization), `episodes.jsonl`,
`diagnostics.jsonl`, `metrics.jsonl`, `result.json`, `progress.json`, and a worker
log. Validate exactly 100 unique ordered seeds, parcel count, configured budget,
actual action-call count, finite values, rate/count ranges and agreement of
aggregates with the episode rows. Failed stages retain partial local records;
there is no automatic resume into an old experiment directory.

Compute each protocol separately:
`weighted_proxy = 0.2 * Easy + 0.3 * Medium + 0.5 * Hard`, using sort accuracy.
All three complete levels are required; no missing-level renormalization. This
is not an official score. No confidence intervals are reported. If added later,
resample **episodes as clusters**, never individual parcels as independent trials.

## Checkpoint input contract

`docs/VALIDATION_INPUT.example.json` is the full schema. Copy it to a local input
file and replace paths with accessible checkpoint locations. Easy and Medium
hashes and inference settings are pinned and must match:

- Easy: `c563ab461b4b1ad87f5a3d2d77df8453334eb6d8a3e6a86f20b7b42676352df9`, horizon 8, steps 32.
- Medium: `95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb`, horizon 8, steps 16.
- Hard: pending parent-provided best checkpoint SHA, winner horizon/steps and
  completed training budget **40000**. Best checkpoint iteration may be earlier.

Every level requires `path`, `sha256`, `selection: "best_eval_sort_accuracy"`,
`provenance_label` (a stable label, not a path), `act_horizon`,
`num_inference_steps`, and `training_completed_iterations` (25000 / 30000 / 40000).
Dry-run permits pending Hard fields. `--run` rejects them. The completion field
is an explicit parent attestation from the reviewed run summary; a best
checkpoint alone does not prove final training completion.

Inside bounded workers, verify source SHA before loading trusted checkpoints.
Check RGB / state_dim 26 / act_dim 4, `config.demo_paths` level tokens and
`config.max_episode_steps`; do not trust legacy `args.num_parcels`. Missing,
wrong-level or mixed-level demo lineage and missing/excessive training budgets
fail. `ema_agent`, or a verified best EMA under `model`, is accepted; raw `agent`
fallback is rejected. Floating source weights must already be float32 and all
weights finite. No fp16 conversion can satisfy the exact identity contract.

## Local execution and independent concurrency

Run only after Hard is done. No notebook automation, install, auth access,
training launch or upload is performed by these commands. Use the current
validated Colab Python, `/content/marso-py312/bin/python`, in
`/content/berlin-marso-hackathon`.

```bash
cd /content/berlin-marso-hackathon
/content/marso-py312/bin/python tools/run_generalization.py --manifest /content/validation-input.json --dry-run > /content/validation-plan.json
/content/marso-py312/bin/python tools/build_repro_package.py --manifest /content/validation-input.json --dry-run > /content/package-plan.json
/content/marso-py312/bin/python tools/run_generalization.py --manifest /content/validation-input.json --out /content/marso-validation --run
/content/marso-py312/bin/python tools/build_repro_package.py --manifest /content/validation-input.json --out /content/marso-validation --run
```

A run creates a **new** `validation_<UTC timestamp>-<nanoseconds>` or
`release_<UTC timestamp>-<nanoseconds>` directory under `--out`; existing run
folders are never reused. `--out` is a local parent directory, not a Drive path.
Dry-run emits the complete plan and performs no GPU/process launch or writes.

GPU evaluation, clean-extraction smoke and the parent-directed Hard worker share
the machine contract `/tmp/marso-active-gpu.lock`, independent of checkout,
extraction, output and experiment names. Each GPU supervisor must acquire an
exclusive nonblocking flock *before* CUDA work and pass the same open descriptor
to its workers for their entire lifetime. Hard may also retain its optional
repo/experiment locks; those cannot replace the machine lock. Never unlink the
machine lock or explicitly unlock it while an inherited worker may survive.
Closing the supervisor's descriptor leaves the worker's inherited lock intact.

The evaluation and smoke supervisors validate and pass the descriptor through
`pass_fds` and `MARSO_GPU_LOCK_FD`. Both actual worker functions verify it before
runtime imports: a regular inode must match the machine lock path, an independent
shared-lock probe must be blocked, and the inherited descriptor must own the
exclusive flock. Missing, stale, unrelated, unlocked or separately opened FDs
fail closed, including direct internal `--worker` invocations. The FD environment
variable alone grants no access. CPU packaging does not require a GPU lock.

A second preflight rejects Hard supervisors (`run_colab_hard.py` and other
`run_colab_*` scripts), train/eval and validation/package supervisor PIDs, plus any
existing `nvidia-smi` compute client, including an idle notebook retaining CUDA
memory. Lock exclusion is independent of launch order for cooperating workers;
the snapshot cannot protect against code that ignores the machine contract.
The Hard sibling's adoption and live CUDA lifetime remain integration gates for
independent review. This lane does not import or modify unpublished Hard code.
No automatic killing of other lanes is performed.

Each owned stage has a finite wall deadline (default evaluation 7200 seconds;
build/smoke 1800 seconds, maximum override 43200). The supervisor persists and
prints progress every 30 seconds, terminates its own process group on timeout,
interrupt, SIGTERM or failure, and records failure. It never stops other lanes.
Total default evaluation worker budget is at most six × 7200 seconds plus
preflight/cleanup. There is no training or runtime launch during implementation.

Results are written locally first and sealed in `ARTIFACT_SHA256.json`, including
partial failed attempts. Parent copies the finished experiment to
`Drive/MyDrive/marso/validations/<exp>` and compares every listed file SHA plus
the seal file's own SHA. Only after that verification may the parent write a
separate copy receipt containing source/Drive destination, seal SHA and verified
file count. These scripts do not duplicate the Hard lane's durability uploader;
`drive_copy` remains pending until the parent supplies that proof. Raw local
plans/logs can contain Colab input paths; only sanitized explicit release
artifacts belong in the release archive.

## Reproducible local release

The builder uses `tools/strip_ckpt.py` with explicit overrides and `fp16=False`,
then sanitizes provenance. It checks tensor dtype, shape and **exact bytes**
(including signed zero) against best EMA before and after the final save/load.
The compact source field is a stable provenance label. Deployment config keeps
needed settings; demo lineage becomes `demos/<level>/<basename>`, with no personal
absolute Mac paths. Optimizer, RNG and arbitrary trainer arguments are omitted.

Outputs include `checkpoints/rgb_dp_{easy,medium,hard}.pt`, `submission.yaml`
with `rgb.policy: warehouse_sort.il_policy:load_dp_rgb` and no override kwargs,
`MODEL_PROVENANCE.json` (source/compact SHA, iteration, parent-completed budget,
config overrides), source hashes, exact installed dependency metadata, declared
pixi files, README/REPRODUCE, and license metadata/existing license files.

The allowlist is explicit in `tools/validation_common.py`: environment/loader,
config YAMLs, DP modules including the reviewed `train_rgbd.py` import dependency,
shared eval helpers, and these tools/docs. It copies working-tree bytes, including
reviewed dirty overrides, not `git archive HEAD`. It excludes `.git`, auth,
`.env`, virtual environments, demos, caches, notebooks and raw outputs. No tests
or training data are required to import and smoke-test the released policy.
No original source license was present in this snapshot; the builder does not
invent a license or claim redistribution permission.

Generated artifacts are included **only** through an explicit manifest list:

```json
{"path":"/content/marso-validation/validation_<exp>/summary.json",
 "sha256":"<64 lowercase hex characters>", "name":"validation_summary.json"}
```

Place that object in top-level `artifacts`. Empty `[]` is valid. Review optional
artifact contents before listing them: path allowlisting and SHA verification
do not remove private text or pixels. The builder copies those bytes exactly;
raw plans/logs are not sanitized by inclusion. Source and copied artifact hashes
must agree. The archive uses sorted files, fixed tar ownership,
mode and timestamps, and fixed gzip metadata. It is extracted and SHA-verified
locally before completion. Reproducible means identical source/checkpoint /
artifact bytes and dependency snapshot yield identical archive bytes; outputs
and timings from separate GPU evaluations need not be byte-identical.

See `docs/REPRODUCE.md` for clean-folder smoke and installation instructions.
A passing local CPU test does not prove GPU loading, actual reset perturbations,
600 evaluated episodes, fresh-machine installation, Drive durability or an
official evaluation. All those proofs remain separate.
