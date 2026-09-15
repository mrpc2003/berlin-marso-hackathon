"""Add a hash-guarded source update cell to the continuation notebook."""
import ast
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHANGED=['eval.py','tools/rollout_logger.py','il/baselines/diffusion_policy/diffusion_policy/evaluate.py']
ADDED=['tests/test_policy_kwargs_config.py','tests/test_diffusion_evaluate_metrics.py']
payload=[]
for name in CHANGED+ADDED:
    content=(ROOT/name).read_text()
    old=subprocess.check_output(['git','show',f'6a7346f:{name}'],cwd=ROOT) if name in CHANGED else None
    payload.append({'path':name,'before_sha256':hashlib.sha256(old).hexdigest() if old else None,
                    'after_sha256':hashlib.sha256(content.encode()).hexdigest(),'content':content})
code='PATCHES = '+repr(payload)+r'''
import hashlib, shutil, time, ast
from pathlib import Path
assert str(REPO)=='/content/berlin-marso-hackathon'
backup=ROOT/'backups'/('eval-fix-'+time.strftime('%Y%m%d-%H%M%S'))
# Preflight every target before any write.
for item in PATCHES:
    rel=Path(item['path']); assert not rel.is_absolute() and '..' not in rel.parts
    dst=REPO/rel
    actual=hashlib.sha256(dst.read_bytes()).hexdigest() if dst.exists() else None
    assert actual in (item['before_sha256'],item['after_sha256']),(str(rel),actual)
    ast.parse(item['content'])
for item in PATCHES:
    dst=REPO/item['path']
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==item['after_sha256']: continue
    if dst.exists():
        saved=backup/item['path']; saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(dst,saved)
    dst.parent.mkdir(parents=True,exist_ok=True)
    temp=dst.with_name(dst.name+'.hermes-tmp')
    temp.write_text(item['content']); temp.replace(dst)
    assert hashlib.sha256(dst.read_bytes()).hexdigest()==item['after_sha256']
run([PY,'-m','pytest','tests/test_policy_kwargs_config.py','tests/test_diffusion_evaluate_metrics.py','-q'],
    'eval_regression_tests')
print('EVAL_PATCH_VERIFIED', {x['path']:x['after_sha256'] for x in PATCHES})
'''
ast.parse(code)
p=ROOT/'MARSO_CURSOR_CONTINUE.ipynb'
nb=json.loads(p.read_text())
bak=ROOT.parent/'.hermes-backups'/('continue-before-eval-fix-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'.ipynb')
shutil.copy2(p,bak)
assert all(c.get('id')!='eval-fix' for c in nb['cells'])
nb['cells'].insert(0,{'cell_type':'code','id':'eval-fix','metadata':{},'execution_count':None,'outputs':[],
                    'source':code.splitlines(keepends=True)})
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('UPDATED',p,'BACKUP',bak)
