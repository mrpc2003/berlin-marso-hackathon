import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cursor_nb', ROOT/'tools/make_cursor_s1_notebook.py')
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_deterministic_notebook_and_embedded_syntax():
    nb=m.build_notebook()
    assert nb==m.build_notebook()
    m.validate_notebook(nb)
    assert len(nb['cells'])==6
    assert len({c['id'] for c in nb['cells']})==6
    for c in nb['cells']:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
            assert c['execution_count'] is None and c['outputs']==[]


def test_bootstrap_seed_and_no_credentials():
    assert "'--seed'" in m.SETUP
    assert "'3.12'" in m.SETUP
    assert 'rglob' not in m.DATA_SETUP
    text=json.dumps(m.build_notebook())
    for forbidden in ['userdata', 'GH_TOKEN', 'KAGGLE_API_TOKEN', 'git push', 'competitions submit']:
        assert forbidden not in text
    assert "'obs/sensor_data/scene_camera/rgb'" in m.DATA_PROBE
    assert "'obs/agent/qpos'" in m.DATA_PROBE
    assert "obs/extra/tcp_pose" in m.DATA_PROBE


def test_failure_output_is_saved(tmp_path):
    ns={'Path':Path,'ROOT':tmp_path,'REPO':ROOT,'ENV':dict(os.environ),'RUN_ID':'test',
        'subprocess':subprocess}
    node=next(x for x in ast.parse(m.SETUP).body if isinstance(x,ast.FunctionDef) and x.name=='run')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'run','exec'),ns)
    with pytest.raises(RuntimeError,match='exit=3'):
        ns['run']([sys.executable,'-c',"print('failure-evidence'); raise SystemExit(3)"],'failure')
    assert (tmp_path/'logs/test_failure.log').read_text()=='failure-evidence\n'


def test_cli_generates_same_content(tmp_path):
    p=tmp_path/'test.ipynb'
    subprocess.run([sys.executable,str(ROOT/'tools/make_cursor_s1_notebook.py'),'--output',str(p)],check=True)
    assert json.loads(p.read_text())==m.build_notebook()
