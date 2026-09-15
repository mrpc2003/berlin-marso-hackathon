"""Failure-injection tests; no real Drive or GPU access."""
import json
from pathlib import Path
import subprocess
import pytest
import tools.run_colab_medium_resume as r


def test_sync_guard_precedes_any_remote_write(tmp_path, monkeypatch):
    local, remote = tmp_path/'local', tmp_path/'remote'
    local.mkdir()
    (local/'result.json').write_text('{}')
    def unmounted(*args):
        raise RuntimeError('mount disappeared')
    monkeypatch.setattr(r, '_drive_guard', unmounted)
    with pytest.raises(RuntimeError, match='mount disappeared'):
        r.sync_tree_once(local, remote)
    assert not remote.exists()


def test_worker_rejects_unapproved_remote_scope():
    with pytest.raises(ValueError, match='remote mirror scope'):
        r._drive_guard('/content/marso-resume/medium_resume_test', '/content/drive/MyDrive/marso/ckpts')


def test_internal_requests_and_inputs_are_not_mirrored(tmp_path, monkeypatch):
    monkeypatch.setattr(r, '_drive_guard', lambda *args: None)
    local, remote = tmp_path/'local', tmp_path/'remote'
    for rel in ['run/.sync/request.json','inputs/latest.pt','run/status.json','ckpts/latest.pt.tmp.123']:
        p=local/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    result=r.sync_tree_once(local,remote)
    assert result['copied']==['run/status.json']
    assert not (remote/'run/.sync').exists()
    assert not (remote/'inputs').exists()


def test_guard_checked_again_before_atomic_replace(tmp_path):
    src,dest=tmp_path/'src',tmp_path/'remote/dest'
    src.write_text('new');dest.parent.mkdir();dest.write_text('old')
    calls=[]
    def guard():
        calls.append(True)
        if len(calls)>1:raise RuntimeError('unmounted during copy')
    with pytest.raises(RuntimeError):r.atomic_copy_verified(src,dest,guard=guard)
    assert dest.read_text()=='old'
    assert not list(dest.parent.glob('*.tmp.*'))


def test_final_sync_rejects_growing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(r, '_drive_guard', lambda *args: None)
    local=tmp_path/'local';local.mkdir();(local/'train.log').write_text('changing')
    monkeypatch.setattr(r,'atomic_copy_verified',lambda *a,**kw:{'status':'skipped_growing_log','sha256':'x'})
    assert not r.sync_tree_once(local,tmp_path/'remote',final=True)['ok']


def test_durability_receipt_last_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(r, '_drive_guard', lambda *args: None)
    local,remote=tmp_path/'local',tmp_path/'remote'
    for rel in ['run/summary.json','ckpts/final.pt','run/diagnostics.json','run/evaluation_results.json','run/status.json']:
        p=local/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    result=r.sync_tree_once(local,remote,final=True)
    receipt=r.finalize_durability(local,remote,result)
    assert receipt['remote_verified']
    assert 'run/status.json' not in receipt['files']
    assert json.loads((remote/'run/status.json').read_text())['status']=='completed'
    for rel,entry in receipt['files'].items():
        assert r.sha(remote/rel)==entry['sha256']
    (remote/'run/diagnostics.json').write_text('changed')
    with pytest.raises(RuntimeError,match='hash differs'):
        r.finalize_durability(local,remote,result)


def test_no_receipt_on_failed_sync(tmp_path):
    with pytest.raises(RuntimeError,match='failed sync'):
        r.finalize_durability(tmp_path/'local',tmp_path/'remote',{'ok':False})
    assert not (tmp_path/'remote').exists()


def test_sync_worker_timeout_is_not_training_failure(tmp_path,monkeypatch):
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('sync',1)
    monkeypatch.setattr(r.subprocess,'run',timeout)
    result=r.run_sync_worker(tmp_path/'local',tmp_path/'remote',{},timeout=1)
    assert not result['ok'] and 'timed out' in result['errors'][0]['error']
    assert not list((tmp_path/'local/run/.sync').glob('*.json'))


def test_stage_survives_reported_sync_failure(tmp_path):
    import sys,os
    calls=[]
    def sync(**kwargs):calls.append(True);return {'ok':False}
    r.run_stage([sys.executable,'-c','print("actual child exit success")'],tmp_path/'train.log',30,lambda **kw:None,os.environ.copy(),tmp_path,sync=sync)
    assert calls and 'actual child exit success' in (tmp_path/'train.log').read_text()
