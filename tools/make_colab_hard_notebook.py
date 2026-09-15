"""Generate an unexecuted, staged Hard notebook from reviewed worktree bytes."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from textwrap import dedent

from tools import run_colab_hard as hard

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'MARSO_COLAB_HARD.ipynb'

MOUNT = '''
import os
from pathlib import Path
from google.colab import drive
if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')
assert os.path.ismount('/content/drive'), 'A real Drive mount is required'
assert Path('/content/drive/MyDrive/marso').is_dir(), 'Existing marso Drive root is required'
print('Drive mounted. Parent stages Hard demos separately; a Drive Hard marker is not required.')
'''

SETUP = '''
import base64, hashlib, json, os, subprocess, time
from pathlib import Path
APPLY_HARD_SETUP = False
if APPLY_HARD_SETUP:
    # This namespace executes only the hash-checked, embedded stdlib Hard bootstrap.
    entry = EMBEDDED_SOURCES['tools/run_colab_hard.py']
    payload = base64.b64decode(entry['content_b64'], validate=True)
    assert hashlib.sha256(payload).hexdigest() == entry['sha256']
    hard_setup = {'__name__': 'hard_bootstrap'}
    exec(compile(payload, 'embedded/tools/run_colab_hard.py', 'exec'), hard_setup)
    EXP = 'hard_' + time.strftime('%Y%m%d_%H%M%S')
    PLAN = hard_setup['bootstrap'](EMBEDDED_SOURCES, EXP)
    print('Prepared new run:', PLAN)
    print('Reviewed originals backed up under source-backups; snapshots under sources.')
else:
    print('Setup skipped. Review this cell, then set APPLY_HARD_SETUP=True and execute only this cell.')
'''

HELPER = '''
# No kernel restart, clone, package installation, environment replacement, or data download.
REPO = Path('/content/berlin-marso-hackathon')
PY = '/content/marso-py312/bin/python'
ENV = dict(os.environ, DISPLAY='', PYOPENGL_PLATFORM='egl', HDF5_USE_FILE_LOCKING='FALSE',
           PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
def hard_phase(phase):
    assert 'PLAN' in globals(), 'Run the reviewed setup cell first (or restore the exact saved PLAN path)'
    subprocess.run([PY, '-u', str(REPO/'tools/run_colab_hard.py'), str(PLAN), '--phase', phase],
                   cwd=REPO, env=ENV, check=True)
'''

CHECKS = '''
RUN_CHECKS = False
if RUN_CHECKS:
    hard_phase('checks')
else:
    print('Checks skipped. After parent stages local h5/json, enable only this cell.')
'''
SMOKE = '''
RUN_SMOKE = False
if RUN_SMOKE:
    hard_phase('smoke')
else:
    print('Smoke skipped. Enable after checks_passed: 100 iterations, all 200 demos, separate weights.')
'''
LAUNCH = '''
RUN_HARD = False
if RUN_HARD:
    hard_phase('run')
else:
    print('Full training skipped. Review smoke_passed, then enable only this cell.')
    print('40,000 iterations from fixed seed 1; no smoke checkpoint resume. Six-hour train deadline.')
'''
MONITOR = '''
# Read-only. Set the exact PLAN path after reconnect; this cell never chooses or launches a run.
from pathlib import Path
import json
PLAN_PATH = str(PLAN) if 'PLAN' in globals() else ''
if PLAN_PATH:
    root = Path(PLAN_PATH).parent.parent
    for rel in ('run/status.json', 'run/summary.json', 'run/durability.json', 'run/train.log'):
        path = root / rel
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 5000))
                print(rel, stream.read().decode(errors='replace'))
else:
    print('Set PLAN_PATH to /content/marso-hard/hard_<timestamp>/run/plan.json. Do not relaunch blindly.')
'''
FINALIZE = '''
RETRY_FINAL_SYNC = False
if RETRY_FINAL_SYNC:
    hard_phase('finalize')
else:
    print('Optional: after compute_status=completed and remounting Drive, retry durability only.')
'''


def source_bundle(repo=REPO):
    bundle = {}
    for rel in sorted(hard.SOURCE_FILES):
        path = hard.child(Path(repo), rel)
        data = path.read_bytes()
        if path.suffix == '.py':
            ast.parse(data, filename=rel)
        old = subprocess.run(['git', 'show', f'HEAD:{rel}'], cwd=repo, capture_output=True, check=False)
        bundle[rel] = dict(base_sha256=hashlib.sha256(old.stdout).hexdigest() if old.returncode == 0 else None,
                           sha256=hashlib.sha256(data).hexdigest(), content_b64=base64.b64encode(data).decode('ascii'))
    hard.validate_source_set(bundle)
    return bundle


def cell(source, name, *, markdown=False):
    text = dedent(source).strip() + '\n'
    result = dict(cell_type='markdown' if markdown else 'code', id=name, metadata={}, source=text.splitlines(True))
    if not markdown:
        ast.parse(text, filename=name)
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    bundle = source_bundle()
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
              cells=[cell('''
# Hard RGB Diffusion Policy — staged local-first run
Execute each stage separately; **do not use Run All**. All action switches start False.
Reuse `/content/berlin-marso-hackathon` and `/content/marso-py312/bin/python`.
Parent stages `il/demos/hard/trajectory.rgb.pd_ee_delta_pos.physx_cuda.{h5,json}`.
This notebook does not install packages or restart the kernel.

1. Mount check; inspect expected Drive root.
2. Enable reviewed source setup once. Every destination must match its base/current hash;
   unknown sources and existing output directories are rejected. Originals are backed up.
3. Enable dataset/CUDA/reference checks. The reference records valid Hard failures honestly.
4. Enable 100-step CUDA smoke. Inspect finite updates and `smoke_passed`.
5. Enable `RUN_HARD` only after reviewing smoke. Fresh seed 1, 40,000 iterations, AMP,
   batch 128, lr 1e-4, ResNet18 scene RGB, horizons 2/8/16; latest saved every 1,000.
   Training eval: 32 episodes / 8 envs every 5,000. Six-hour training limit.
6. Monitor by exact saved plan path. Re-executing a training phase is rejected.

Outputs: `/content/marso-hard/hard_<timestamp>` and
`MyDrive/marso/recoveries/hard/hard_<timestamp>`. Local completion and remote pending are separate.
The best sort-accuracy checkpoint is evaluated on four local seed-5000 configurations, then
recorded with one video env and eight diagnostic episodes. Scores are local validation,
not official or heldout. Shared eval/diagnostic loops execute 799 steps with an 800-step limit.
A missing best checkpoint (all tracked sort metrics zero) uses final with an explicit fallback record.
''', 'instructions', markdown=True),
                     cell(MOUNT, 'mount-check'),
                     cell('EMBEDDED_SOURCES = ' + repr(bundle) + '\n' + dedent(SETUP), 'reviewed-setup'),
                     cell(HELPER + CHECKS, 'dataset-gpu-reference'),
                     cell(SMOKE, 'smoke-gate'), cell(LAUNCH, 'full-launch'),
                     cell(MONITOR, 'monitor'), cell(FINALIZE, 'retry-durability')])
    validate_notebook(nb)
    return nb


def validate_notebook(nb):
    code = {c['id']: ''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'}
    for name, source in code.items():
        ast.parse(source, filename=name)
    for name, flag in [('reviewed-setup', 'APPLY_HARD_SETUP'), ('dataset-gpu-reference', 'RUN_CHECKS'),
                       ('smoke-gate', 'RUN_SMOKE'), ('full-launch', 'RUN_HARD'), ('retry-durability', 'RETRY_FINAL_SYNC')]:
        tree = ast.parse(code[name])
        assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == flag for t in n.targets)]
        hard.require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Constant)
                     and assignments[0].value.value is False, f'Unsafe launch flag: {flag}')
    tree = ast.parse(code['reviewed-setup'])
    bundle = ast.literal_eval(tree.body[0].value)
    hard.validate_source_set(bundle)
    current = source_bundle()
    hard.require(bundle == current, 'Stale embedded sources; regenerate from this worktree')
    for rel, entry in bundle.items():
        payload = base64.b64decode(entry['content_b64'], validate=True)
        hard.require(hashlib.sha256(payload).hexdigest() == entry['sha256'], 'Corrupt embedded payload')
        if rel.endswith('.py'):
            ast.parse(payload, filename='embedded/' + rel)
    for name, source in code.items():
        source = '\n'.join(source.splitlines()[1:]) if name == 'reviewed-setup' else source
        hard.require('force_remount' not in source and 'pip install' not in source and 'os.kill(' not in source,
                     'Unsafe notebook operation')
    hard.require(all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells']),
                 'Saved notebook must be unexecuted')
    # Generation uses the system Python with existing nbformat. CPU unit tests can also
    # validate all source/AST invariants without adding nbformat to the training test venv.
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(nb)


if __name__ == '__main__':
    import nbformat
    notebook = build_notebook()
    nbformat.validate(notebook)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(OUTPUT)
