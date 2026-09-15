"""Static contract tests for the credential-free recovery notebook."""
import ast
import base64
import hashlib
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

import tools.make_colab_medium_resume_notebook as gen
import tools.run_colab_medium_resume as resume


def test_each_runtime_fragment_is_valid_python():
    for src in (gen.SETUP, gen.RUNTIME_SETUP, gen.STAGE, gen.CHECKPOINT_PROBE, gen.SIM, gen.LAUNCH, gen.MONITOR):
        ast.parse(dedent(src))


def test_setup_logs_local_and_checks_real_drive():
    assert "os.path.ismount('/content/drive')" in gen.SETUP
    assert "log = LOCAL_SETUP/" in gen.SETUP
    assert "log = ROOT/" not in gen.SETUP
    assert "saved=LOCAL_SETUP/'source-backups'" in gen.SETUP
    assert "if 'it/s' not in line" not in gen.SETUP
    assert "rc = p.wait()" in gen.SETUP
    assert "if rc: raise RuntimeError" in gen.SETUP


def test_launch_default_off_does_not_need_kernel_globals(capsys):
    exec(compile(dedent(gen.LAUNCH), '<launch>', 'exec'), {})
    assert 'Not started.' in capsys.readouterr().out
    assert 'start_new_session' not in gen.LAUNCH
    assert "run([PY,'-u'" in gen.LAUNCH


def test_fixed_recovery_input_and_full_training_schedule():
    assert gen.CHECKPOINT_SHA == resume.EXPECTED_RESUME_SHA256 == '8082cbb5b4da6a871ad754266d13a8efad714790e6b956502f8f440b328a205b'
    assert gen.PRIOR_BEST_SHA == resume.EXPECTED_PRIOR_BEST_SHA256 == '95ecf9a9c28153aef6ceb130e7a571897806a44d2196a34bade7a5f1f811fdeb'
    assert gen.PRIOR_EXP == 'medium_resume_20260914-040647'
    assert gen.DRIVE_INPUT_SOURCE == 'gdrive:marso/recoveries/medium/medium_resume_20260914-040647/ckpts'
    assert "'total_iters':30000" in gen.STAGE
    assert "resume['iteration']==29000" in gen.CHECKPOINT_PROBE
    assert "best['iteration']==25000" in gen.CHECKPOINT_PROBE
    assert "best['best_eval_metrics']['sort_accuracy']==1.0" in gen.CHECKPOINT_PROBE
    assert 'validate_medium_checkpoint(resume, expected_iteration=29000)' in gen.CHECKPOINT_PROBE
    assert 'validate_prior_best_checkpoint(best)' in gen.CHECKPOINT_PROBE
    assert "'optimizer'" in gen.CHECKPOINT_PROBE and "'lr_scheduler'" in gen.CHECKPOINT_PROBE
    assert 'torch.isfinite' in gen.CHECKPOINT_PROBE
    assert 'old_ckpts/filename' in gen.STAGE
    assert 'target = INPUTS/filename' in gen.STAGE
    assert "old_ckpts = ROOT/'recoveries/medium'/OLD_EXP/'ckpts'" in gen.STAGE
    assert "'drive_input_source':DRIVE_INPUT_SOURCE" in gen.STAGE
    assert "'prior_exp':OLD_EXP" in gen.STAGE
    assert 'load_plan(PLAN)' in gen.STAGE
    assert 'atomic_copy_verified(source, target, guard=lambda: ensure_remote_ready(ROOT))' in gen.STAGE
    assert "EXP = f'medium_resume_29000_{RUN_ID}'" in gen.STAGE
    assert 'assert not LOCAL_ROOT.exists()' in gen.STAGE
    assert "assert not (ROOT/'recoveries/medium'/EXP).exists()" in gen.STAGE


@pytest.mark.parametrize('existing,valid', [(True, True), (True, False), (False, False)])
def test_runtime_reuses_only_a_fully_valid_environment(tmp_path, capsys, existing, valid):
    py = tmp_path/'bin/python'
    if existing:
        py.parent.mkdir()
        py.write_text('test placeholder; never executed')
    calls = []

    def fake_run(cmd, label):
        calls.append((cmd, label))
        if label == 'reuse_probe' and not valid:
            raise RuntimeError('invalid environment')
        return 'validated'

    scope = {'Path': Path, 'PY': str(py), 'VENV': tmp_path, 'REPO': gen.REPO,
             'RUN_ID': 'test', 'sys': SimpleNamespace(executable='/kernel/python'), 'run': fake_run}
    exec(compile(dedent(gen.RUNTIME_SETUP), '<runtime-setup>', 'exec'), scope)
    labels = [label for _, label in calls]
    if valid:
        assert labels == ['reuse_probe']
        assert 'Reusing validated Python 3.12 environment' in capsys.readouterr().out
    else:
        assert labels == (['reuse_probe'] if existing else []) + ['uv', 'uv_python'] + ([] if existing else ['venv']) + ['torch', 'deps', 'editable', 'cuda_probe']
    probe = scope['probe']
    ast.parse(probe)
    for check in ('3.12.3', '2.11.0', '0.26.0', '3.0.1', '1.3.0', '0.38.0', '0.1.1',
                  'torch.cuda.is_available()', 'torch.isfinite(y).all()', "version('numpy')", 'EMAModel'):
        assert check in probe
    assert 'reuse_probe' not in gen.base.SETUP


def test_built_notebook_schema_ast_output_name_and_embedded_sources():
    nbformat = pytest.importorskip('nbformat', reason='nbformat is required to build and validate notebooks')
    nb = gen.build_notebook()
    nbformat.validate(nb)
    assert gen.OUTPUT == gen.REPO/'MARSO_COLAB_MEDIUM_RESUME_29000.ipynb'
    assert [cell['id'] for cell in nb['cells']] == [
        'resume-drive', 'resume-setup', 'resume-inputs', 'resume-reference', 'resume-launch', 'resume-monitor']
    cells = {cell['id']: ''.join(cell['source']) for cell in nb['cells']}
    for cell in nb['cells']:
        ast.parse(''.join(cell['source']))
        assert cell['execution_count'] is None and cell['outputs'] == []
    assert 'RUN_RESUME = False' in cells['resume-launch']
    assert 'iteration 29000; target remains 30000' in cells['resume-launch']
    for constant in (gen.CHECKPOINT_SHA, gen.PRIOR_BEST_SHA, gen.PRIOR_EXP, gen.DRIVE_INPUT_SOURCE):
        assert constant in cells['resume-inputs']
    embedded = ast.literal_eval(ast.parse(cells['resume-setup']).body[0].value)
    for rel, item in embedded.items():
        data = (gen.REPO/rel).read_bytes()
        assert base64.b64decode(item['content_b64']) == data
        assert item['sha256'] == hashlib.sha256(data).hexdigest()
    assert "kwargs['num_parcels']==4" in cells['resume-reference']
    for stale in ('978b8e10ac5d91753e7c7d337c1dc797d4a5b17a6f2f2fa5229adf66dc3857a4',
                  'c0be440c121ac15aaa4cd67e185a2a1f58895a69f5c1c37155b13fb048d9844c',
                  'medium_20260914-014131', 'iteration 22000'):
        assert stale not in json.dumps(nb)


def test_checkpoint_probe_rejects_wrong_hash_before_loading(tmp_path, monkeypatch):
    import sys
    import torch
    latest, best = tmp_path/'latest.pt', tmp_path/'best_eval_sort_accuracy.pt'
    latest.write_bytes(b'wrong'); best.write_bytes(b'wrong')
    monkeypatch.setattr(sys, 'argv', ['probe', str(latest), str(best)])
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('must hash before loading'))
    with pytest.raises(AssertionError):
        exec(compile(dedent(gen.CHECKPOINT_PROBE), '<checkpoint-probe>', 'exec'), {})


def test_four_32_episode_settings_and_diagnostics_remain():
    source = Path(resume.__file__).read_text()
    assert 'for horizon, steps in ((8, 16), (4, 16), (8, 32), (4, 32)):' in source
    assert '_eval_cfg(cfg32, 32)' in source
    assert 'rows[0]["requested_n_episodes"] == 32' in source
    assert 'tools/rollout_logger.py' in source and '_eval_cfg(cfg8, 8)' in source
    assert 'final_sync = sync(final=True)' in source


def test_no_credentials_publishing_or_old_training_launch():
    sources='\n'.join((gen.SETUP,gen.STAGE,gen.LAUNCH,gen.MONITOR))
    for forbidden in ('KAGGLE_API_TOKEN','GH_TOKEN','competitions submit','git push','flags.total_iters=8000'):
        assert forbidden not in sources
    assert 'tools/run_colab_medium_resume.py' in gen.LAUNCH
    assert 'tools/run_colab_medium.py' not in gen.LAUNCH
