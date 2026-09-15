# Fresh frozen600 validation with durable partial evidence

## Recovery boundary

Parent reports that the old Colab runtime `MarsoT4b` was removed. Its last UI
log showed Easy 100, Medium 100, Hard 96; no completed 600-episode validation
reached Drive. Those UI counts are historical partial progress, not a final
validation result. All three trained best checkpoints and the complete Hard
artifacts are preserved according to parent. This implementation has not
reopened or verified those remote objects.

Parent provisions a fresh T4, restores the **same versions and reviewed source
bytes**, and supplies the verified three-checkpoint manifest. Run all 600
episodes from the beginning, with the frozen protocol and existing winner
settings. Do not train again, tune, substitute checkpoints, resume an old
experiment, or infer the missing result from the last UI log.

The protocol file remains `docs/VALIDATION_PROTOCOLS.json`, SHA-256:

```text
e480d74f53c7aa963d6058624c0f590e0f6fbdccbe2effad061760b958878507
```

`tools/run_validation_durable.py` uses only the standard library plus the
existing CPU-safe validation helpers. The approved CLI, worker seed, batch
validation, machine GPU lock and source allowlist remain unchanged. The new
wrapper, its tests and this document are outside that allowlist. Its filename
does not match `check_idle_gpu`'s rejection pattern for an active parent.

## Exact invocation

Run from the restored reviewed checkout. These commands assume parent has
placed the fully verified manifest at `/content/validation_inputs.json` and
restored the interpreter at the previously used path. These are deployment
paths to prepare, not claims that a new runtime already exists.

```bash
# Preflight only: no GPU work and no persistent output directories.
/content/marso-py312/bin/python -B tools/run_validation_durable.py \
  --manifest /content/validation_inputs.json \
  --out /content/marso-durable \
  --drive /content/drive/MyDrive/marso/validations \
  --timeout 14400 --sync-interval 30 --sync-timeout 60

# Fresh exact600 evaluation; no training or settings override.
/content/marso-py312/bin/python -B tools/run_validation_durable.py \
  --manifest /content/validation_inputs.json \
  --out /content/marso-durable \
  --drive /content/drive/MyDrive/marso/validations \
  --timeout 14400 --sync-interval 30 --sync-timeout 60 --run
```

There is no command override, worker mode, mount bypass or resume flag.
All three path flags are required. Without `--run`, strict preflight still
requires an **actual mount** at `/content/drive`, a private destination below
`/content/drive/MyDrive`, and present, hash-matching, nonpending checkpoints.
A directory that merely has the name `drive` does not pass. `MyDrive` is a
namespace check; parent supplies an unshared private folder. Manifest paths
and all checkpoint/artifact path components reject symlinks and `..`. Relative
manifest references resolve against the manifest's directory. Input paths may
be on Drive; preflight hashes run in a bounded subprocess. If necessary,
increase `--sync-timeout` for large restored checkpoints before starting.

`--out` must be local and outside Drive; the approved CLI also rejects any
component named `drive`. Timers accept integer seconds from 1 through 43200.
Defaults are a 14400-second compute deadline, 30-second copy-start interval,
and 60-second timeout for each preflight/copy job. Only one copy job runs at a
time; a slow cycle delays the next start. The overall wall bound additionally
includes preflight, termination cleanup (25-second supervisor grace and
bounded kill/reap waits), and one final copy attempt. It is not a promise that
600 episodes fit within the default four hours.

## Ownership and result paths

Every run creates a unique `durable_<UTC>_<UUID>` directory locally and an
exclusively created directory of the same name in the supplied private Drive
parent. Existing run directories are never reused. A failed preflight can
leave a new local record directory with `compute_status: not_started`.

```text
LOCAL_PARENT/durable_<id>/
  records/
    invocation.json       # exact approved child argv, source/wrapper hashes
    input_manifest.json   # fixed manifest snapshot, absolute input references
    child.log             # captured approved supervisor stdout + stderr
    status.json           # current local wrapper status
  validation_runs/
    validation_<actual-id>/  # created and sealed by the approved CLI
      plan.json
      fresh_seed_easy/ ... stress_hard/
      summary.json         # only present after approved compute success
      ARTIFACT_SHA256.json

DRIVE_PARENT/durable_<id>/
  records/                 # copied wrapper evidence
  validation_<actual-id>/   # the actual child result, without renaming
  DURABILITY_STATUS.json   # authoritative separate copy receipt
```

Printed JSON and local `records/status.json` identify `validation_dir`,
`validation_result` (the actual `summary.json` when present),
`remote_validation_dir`, and `remote_validation_result` after a successful
copy receipt. A proposed remote result path alone is not proof of durability;
check `remote_status`. The wrapper discovers at most one validation directory
inside its own new, initially empty `validation_runs` parent. It does not
search global outputs or other experiments.

Parent uses these actual result paths with the already approved build,
archive-verification and clean-extraction smoke CLIs, separately. The wrapper
does not build, smoke-test or publish a release. Do not edit the child
`summary.json` field `drive_copy`; the separate durability receipt supplies
that evidence without invalidating the approved seal.

## Copy and completion contracts

Each copy cycle reads only the owned records and discovered child directory.
The file allowlist covers approved plan/preflight/summary/failure records,
six stage directories, worker logs, progress, configurations, metrics,
episode/pose JSONL records and reset PPM images. It excludes `.tmp` and
`.locks`. Unknown directories/files, auth material, demos, checkpoints,
nonregular files and symlinks fail closed and are not copied. Input references
inside plans are metadata only; their referenced files are never traversed.
These private raw records can include Colab paths and are not sanitized
release artifacts.

Copies use unique temporary destination files, `fsync`, atomic replacement,
independent source/temp/destination SHA-256 reads and source inode/size/time
checks. A changed source leaves the previous durable destination intact and
is listed in `transient_files` for a later cycle. An interrupted copy may
leave an ephemeral destination temp file; it is excluded from the payload.
Live JSONL counts include only newline-terminated, valid frozen-seed rows;
an incomplete trailing line is not counted. Complete 100-row stages get
metrics recomputed with the approved aggregation helper, with available
stage result metrics checked against the rows.

Every successful cycle atomically writes and reads back
`DURABILITY_STATUS.json` with the actual validation path, most recently
started stage (`phase`), counts/metrics for all six stages, transient/errors,
and separate `compute_status` and `remote_status`. `stages` reports local
observations; `durable_stages` recomputes counts/metrics from copied Drive
records, which may lag when a source was transient. The receipt is a snapshot;
its timestamp tells how recent the evidence is. The copied
`records/status.json` may lag this receipt.

| Wrapper status / exit | Meaning |
| --- | --- |
| `running` | Computation is active; remote may be pending, partial or failing. |
| `completed` / 0 | Child exited zero, local frozen600 results pass checks, and exact local/Drive payload plus seal SHA match. |
| `pending_durability` / 2 | Child exited zero, but final copy or verification failed/timed out. Never treat as durable completion. |
| `failed` / 1 | Preflight, computation, timeout, interruption or cleanup failed; partial evidence is retained/copied when possible. |
| Dry-run / 0 | Preflight passed only; no evaluation ran. |

`remote_status: partial_verified` means individual stable copies and the
progress receipt succeeded; it is not final validation. Only
`remote_status: final_verified` accompanies durable completion. Final checks
require the approved `ARTIFACT_SHA256.json`, exact payload membership, every
file hash on both sides and the seal file's own SHA. The local payload is
rechecked after the Drive reads. A partial run, missing seal, extra file,
changed source, altered metrics, incomplete episode set or copy failure
cannot produce `completed`.

## Interruptions and remaining limits

The wrapper captures child stdout through a pipe into a local file, and emits
nonblocking heartbeat/status JSON. SIGTERM, SIGHUP, Ctrl-C or an observed
broken notebook output pipe request graceful SIGTERM of the approved
supervisor. That supervisor owns the machine GPU lock and cleanup of its
separately sessioned workers. After the grace period, the wrapper escalates
only to recorded descendants whose PID birth identities still match. It
never selects processes by name, uses a global kill, or stops other lanes.
Linux ancestry is observed through `/proc` while the child runs. If process
inspection fails, the direct child is still stopped and cleanup is reported
as pending rather than completed.

Drive operations, including mount checks, run in a separate bounded process.
Copy errors do not stop computation; a final bounded attempt also runs after
compute failure. If an I/O process remains unkillable in kernel I/O, further
copies are suppressed to avoid overlapping writers. No unbounded join or
retry loop is used. Parent must inspect pending/failed receipts and perform
any later approved manual copy; there is no wrapper resume/reuse mode.

SIGKILL of the wrapper, destruction of the VM, sudden local-disk loss and
loss of unflushed FUSE/server data cannot be repaired by this wrapper. A
notebook disconnect that sends no signal and keeps its output pipe open is
not detectable; the bounded run continues. A worker orphaned before it can
be observed after an abrupt supervisor kill is another limitation. Hash
readback proves the mounted filesystem's returned bytes at that time, not
Google's server persistence after the VM is destroyed. Parent should
independently reopen Drive evidence before releasing the runtime.

## CPU verification

Use the existing CPU environment; install nothing:

```bash
../../.venv-cpu/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_validation_durable.py tests/test_validation_package.py \
  tests/test_policy_kwargs_config.py tests/test_diffusion_evaluate_metrics.py \
  tests/test_il_policy.py
```

Tests exercise real files, atomic copies, tampering, source mutation,
symlinks, partial/final seals, copy timeouts, real subprocesses/signals,
separately sessioned workers, the approved CPU supervisor cleanup, and an
unrelated process that must survive. This macOS sandbox denies `ps`, so
process tests supply the known test-created PID relationships at the
lower-level discovery seam; signals and process cleanup are real. Production
Linux `/proc` ancestry, actual Drive/FUSE behavior, T4/lock inheritance,
restored versions/checkpoints, the 600-episode run and final release smoke
remain live verification gates for parent. CPU tests do not establish them.
