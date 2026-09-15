# Reproduce this local RGB policy release

Use Linux x86_64, Python 3.12, a CUDA-capable NVIDIA GPU and working Vulkan /
SAPIEN support. The parent already validated `/content/marso-py312/bin/python`
on the Colab T4. This release does not install software or include demos,
optimizer state, virtual environments or simulator asset caches.

The declared environment is `pixi.toml` plus `pixi.lock`. The built package's
`PACKAGE_VERSIONS.json` and `requirements-repro.txt` record exact **installed**
versions; inspect differences from pixi before reproducing. Needed libraries
include torch, torchvision, mani-skill, sapien, diffusers, gymnasium, hydra-core,
omegaconf, numpy, transforms3d, tyro, matplotlib and tensorboard (as declared by
pixi), plus h5py and tqdm imported by the RGB loader's vendored training modules.
They are required even though this workflow never trains. The Panda simulator
assets must already be available through the existing ManiSkill environment.

For a separately provisioned environment, the manual starting point is
`pixi install --locked`; ensure h5py and tqdm are available at the recorded
versions, or manually install `requirements-repro.txt` into a Python 3.12
environment with the appropriate CUDA torch distribution. This is installation
guidance, **not a claim that a fresh-machine install was tested**. No automatic
installation is part of the package scripts. Use a clean extraction with the
existing validated dependencies first.

## Clean extraction and GPU smoke

Parent supplies the completed archive path and expected SHA from
`build_result.json`. Use a new empty folder; keep smoke outputs outside it.

```bash
ARCHIVE=/content/marso-validation/release_<actual-exp>/release.tar.gz
sha256sum "$ARCHIVE"
# Compare with build_result.json before extraction.
CLEAN=$(mktemp -d /content/marso-clean-XXXXXX)
tar -xzf "$ARCHIVE" -C "$CLEAN"
cd "$CLEAN"
PYTHONDONTWRITEBYTECODE=1 /content/marso-py312/bin/python tools/build_repro_package.py --verify-only
PYTHONDONTWRITEBYTECODE=1 /content/marso-py312/bin/python tools/build_repro_package.py --smoke --out /content/marso-validation --timeout 1800
```

`--verify-only` checks every archive file against `ARTIFACT_SHA256.json` and
rejects missing, extra, symlinked or changed files. `--smoke` does that again,
checks the GPU lock/process state, imports the submission entrypoint from this
extraction, loads each checkpoint with **no override kwargs**, resets a real
RGB environment once per level at seed 9100, and checks a finite N×4 action
within [-1,1] followed by one environment step. These tiny one-step episodes
prove loading/action execution, not task success. It records the loaded
settings and three checkpoint hashes in a new `smoke_<exp>/smoke_result.json`.
It does not use the original checkout's Python path or write into the archive.
The supervisor and actual smoke worker require the same machine lock
`/tmp/marso-active-gpu.lock` used by validation and the parent-directed Hard
worker. Internal `--worker smoke` calls fail without the verified inherited
descriptor; changing checkout or extraction directories cannot bypass the lock.

The policy entrypoint and relative checkpoint paths are declared in
`submission.yaml`. For a normal rollout the original evaluator can be invoked
from the extraction, for example:

```bash
flock -n /tmp/marso-active-gpu.lock /content/marso-py312/bin/python eval.py difficulty=easy obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb checkpoint=checkpoints/rgb_dp_easy.pt eval_config=conf/eval/default.yaml record_video=false
```

This example uses the existing small evaluation config, not the frozen 100-seed
protocol. Its Linux `flock` wrapper is required for the legacy evaluator's
machine exclusion; that evaluator has no internal worker-lock guard. Reproduce
the independent 600-episode evaluation from the parent
checkout using `docs/VALIDATION_PROTOCOL.md` and the verified best **source**
checkpoint manifest; those full trainer inputs are intentionally not bundled.

`MODEL_PROVENANCE.json` links compact weights to source SHA and winner knobs.
`SOURCE_MANIFEST.json` describes the actual reviewed working-tree bytes.
`PACKAGE_INPUT.json` uses stable labels; optional evaluation artifacts are only
those explicitly named by the parent. Preserve `LICENSES.md` and any included
license texts. The archive is local until the parent performs and verifies its
Drive copy to `marso/validations/<exp>`.
