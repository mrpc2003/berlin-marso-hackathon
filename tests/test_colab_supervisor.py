import os
from pathlib import Path
import runpy
import sys
import pytest

M=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/run_colab_s1.py'))


def test_timestamped_experiment_name_contract():
    import ast
    source=(Path(__file__).resolve().parents[1]/'tools/run_colab_s1.py').read_text()
    tree=ast.parse(source)
    expr=next(n.test for n in ast.walk(tree) if isinstance(n,ast.Assert) and 'exp.replace' in ast.unparse(n.test))
    compiled=compile(ast.Expression(body=expr),'experiment-contract','eval')
    assert eval(compiled,{'exp':'rgb_dp_easy_s1_20260913-154033'})
    assert not eval(compiled,{'exp':'../../other'})
    assert not eval(compiled,{'exp':''})


def test_atomic_json_and_hash(tmp_path):
    p=tmp_path/'state.json'
    M['save_json'](p,{'status':'test'})
    assert p.read_text().endswith('}')
    assert not p.with_name('state.json.tmp').exists()
    assert len(M['sha'](p))==64


def test_stage_saves_real_process_output(tmp_path):
    state={}
    M['run_stage']([sys.executable,'-c',"print('actual-process-ok')"],tmp_path/'ok.log',5,
                   lambda **kw:state.update(kw),dict(os.environ),tmp_path)
    assert (tmp_path/'ok.log').read_text()=='actual-process-ok\n'
    assert state['stage_pid'] is None


def test_stage_rejects_failure(tmp_path):
    with pytest.raises(RuntimeError,match='exited 2'):
        M['run_stage']([sys.executable,'-c','raise SystemExit(2)'],tmp_path/'bad.log',5,
                       lambda **kw:None,dict(os.environ),tmp_path)


def test_timeout_terminates_owned_child(tmp_path):
    state={}
    with pytest.raises(RuntimeError,match='Time limit'):
        M['run_stage']([sys.executable,'-c','import time; time.sleep(10)'],tmp_path/'timeout.log',0.05,
                       lambda **kw:state.update(kw),dict(os.environ),tmp_path)
    with pytest.raises(ProcessLookupError): os.kill(state['stage_pid'],0)
