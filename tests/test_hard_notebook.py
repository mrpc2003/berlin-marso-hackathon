import ast
import base64
import copy
from pathlib import Path

import pytest

from tools import make_colab_hard_notebook as notebook
from tools import run_colab_hard as hard


def test_nb_stages_and_embedded_dirty_sources():
    nb = notebook.build_notebook()
    notebook.validate_notebook(nb)
    ids = [c['id'] for c in nb['cells']]
    assert ids == ['instructions', 'mount-check', 'reviewed-setup', 'dataset-gpu-reference',
                   'smoke-gate', 'full-launch', 'monitor', 'retry-durability']
    source = ''.join(nb['cells'][2]['source'])
    embedded = ast.literal_eval(ast.parse(source).body[0].value)
    for rel in ('eval.py', 'tools/rollout_logger.py', 'il/baselines/diffusion_policy/train_rgbd.py',
                'il/baselines/diffusion_policy/diffusion_policy/evaluate.py',
                'tools/run_colab_medium_resume.py', 'conf/difficulty/hard.yaml', 'il/conf/method/dp_rgb_hard.yaml'):
        assert rel in embedded
        assert base64.b64decode(embedded[rel]['content_b64']) == (notebook.REPO / rel).read_bytes()
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells'])


@pytest.mark.parametrize('cid,flag', [('full-launch', 'RUN_HARD'), ('smoke-gate', 'RUN_SMOKE'),
                                     ('dataset-gpu-reference', 'RUN_CHECKS'), ('reviewed-setup', 'APPLY_HARD_SETUP')])
def test_notebook_unsafe_flags_rejected(cid, flag):
    nb = notebook.build_notebook()
    cell = next(c for c in nb['cells'] if c['id'] == cid)
    cell['source'] = [''.join(cell['source']).replace(flag + ' = False', flag + ' = True')]
    with pytest.raises(ValueError, match='Unsafe launch flag'):
        notebook.validate_notebook(nb)


def test_stale_or_corrupt_embedded_python_rejected():
    nb = notebook.build_notebook()
    cell = next(c for c in nb['cells'] if c['id'] == 'reviewed-setup')
    source = ''.join(cell['source'])
    tree = ast.parse(source)
    bundle = ast.literal_eval(tree.body[0].value)
    bundle['eval.py']['content_b64'] = base64.b64encode(b'broken(\n').decode()
    cell['source'] = ['EMBEDDED_SOURCES = ' + repr(bundle) + '\n' + '\n'.join(source.splitlines()[1:])]
    with pytest.raises(ValueError, match='Stale'):
        notebook.validate_notebook(nb)


def test_no_forced_mount_install_or_restart():
    assert 'force_remount' not in notebook.MOUNT
    assert 'drive.mount' in notebook.MOUNT and 'ismount' in notebook.MOUNT
    for source in (notebook.SETUP, notebook.HELPER, notebook.CHECKS, notebook.SMOKE, notebook.LAUNCH):
        assert 'pip install' not in source and 'force_remount' not in source and 'kill(' not in source


def test_saved_notebook_is_current_unexecuted_and_ast_valid():
    import json
    saved = json.loads(notebook.OUTPUT.read_text())
    notebook.validate_notebook(saved)
    assert saved == notebook.build_notebook()
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in saved['cells'])


def test_secret_scan_only_reports_locations_never_secret_contents():
    from tools.audit_hard_notebook import scan_text
    token = 'gh' + 'p_' + 'a' * 36
    findings = scan_text('fixture.py', 'value = ' + repr(token))
    assert findings == [{'path': 'fixture.py', 'line': 1, 'rule': 'api_token'}]
    assert token not in repr(findings) and 'value =' not in repr(findings)


def test_manifest_is_explicit_and_complete_without_git_inventory(monkeypatch):
    original = notebook.subprocess.run
    monkeypatch.setattr(notebook.subprocess, 'check_output', lambda *a, **kw: pytest.fail('No Git inventory allowed'))
    monkeypatch.setattr(notebook.subprocess, 'run', original)
    bundle = notebook.source_bundle()
    assert set(bundle) == set(hard.SOURCE_FILES)
    assert 'tools/hard_gpu_worker.py' in bundle
    assert 'warehouse_sort/il_policy.py' in bundle
    assert 'il/baselines/diffusion_policy/train_rgbd.py' in bundle
