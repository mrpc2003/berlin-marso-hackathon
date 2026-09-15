"""Append the reviewed launch + read-only monitor cells to the connected notebook."""
import ast, datetime, hashlib, json, shutil, subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
source=(R/'tools/run_colab_s1.py').read_text()
paths=subprocess.check_output(['git','ls-files','--','warehouse_sort','conf','il/conf',
 'il/baselines/diffusion_policy','eval.py','tools/rollout_logger.py'],cwd=R,text=True).splitlines()
paths=[p for p in paths if p.endswith(('.py','.yaml'))]
hashes={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths}
hashes['tools/run_colab_s1.py']=hashlib.sha256(source.encode()).hexdigest()
launch='SUPERVISOR_SOURCE = '+repr(source)+'\nSOURCE_HASHES = '+repr(hashes)+r'''
import json, os, subprocess, time
from pathlib import Path

def launch_s1_main():
    global MAIN_EXP, MAIN_DIR
    preflight=json.loads(SUMMARY.read_text())
    assert preflight['smoke_eval']['n_episodes']==8
    assert preflight['training_it_s']>=4, 'Re-plan the time budget before launch'
    assert json.loads(SIM_REPORT.read_text())['passed']
    MAIN_EXP='rgb_dp_easy_s1_'+preflight['run_id']
    MAIN_DIR=ROOT/'runs'/MAIN_EXP
    MAIN_DIR.mkdir(parents=True,exist_ok=True)
    status=MAIN_DIR/'status.json'
    if status.exists():
        previous=json.loads(status.read_text())
        if previous.get('status')=='completed':
            print('ALREADY_COMPLETED',previous); return
        if previous.get('status') not in ('failed','completed'):
            try: os.kill(previous['pid'],0)
            except ProcessLookupError: pass
            else:
                print('ALREADY_RUNNING',previous); return
    script=REPO/'tools/run_colab_s1.py'
    if script.exists(): assert script.read_text()==SUPERVISOR_SOURCE, 'Different supervisor exists'
    else: script.write_text(SUPERVISOR_SOURCE)
    plan={'repo':str(REPO),'root':str(ROOT),'python':PY,'exp':MAIN_EXP,'total_iters':25000,
          'source_sha256':SOURCE_HASHES,'preflight':str(SUMMARY),'steady_training_it_s':preflight['training_it_s'],
          'reason_for_25k':'Measured 5.23 it/s; ~80 min training plus evaluation, schedule fixed before training',
          'submission_allowed':False,'push_allowed':False}
    plan_path=MAIN_DIR/'launch_plan.json'
    plan_path.write_text(json.dumps(plan,indent=2))
    with (MAIN_DIR/'supervisor.log').open('a') as log:
        p=subprocess.Popen([PY,str(script),str(plan_path)],cwd=str(REPO),env=ENV,
             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1)
    assert p.poll() is None, (MAIN_DIR/'supervisor.log').read_text()[-5000:]
    assert status.is_file(), 'Supervisor did not publish status'
    state=json.loads(status.read_text())
    assert state['pid']==p.pid and state['status']=='training',state
    print('MAIN_TRAINING_STARTED',json.dumps(state,indent=2))

launch_s1_main()
'''
monitor=r'''
import json, os, re, subprocess
from pathlib import Path
main_dir=ROOT/'runs'/('rgb_dp_easy_s1_'+RUN_ID)
status=json.loads((main_dir/'status.json').read_text())
print('STATUS',json.dumps(status,indent=2))
try: os.kill(status['pid'],0); print('SUPERVISOR_ALIVE')
except ProcessLookupError: print('SUPERVISOR_EXITED')
log=Path(status.get('log',main_dir/'train.log'))
if log.exists():
    with log.open('rb') as f:
        f.seek(0,2); end=f.tell(); f.seek(max(0,end-12000)); text=f.read().decode('utf-8','replace')
    lines=[x for x in re.split(r'[\r\n]+',text) if x.strip()]
    print('RECENT_LOG\n'+'\n'.join(lines[-10:]))
subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],check=True)
ckdir=ROOT/'ckpts'/status['exp']
print('CHECKPOINTS',[(p.name,p.stat().st_size) for p in sorted(ckdir.glob('*.pt'))])
if (main_dir/'summary.json').exists(): print('FINAL', (main_dir/'summary.json').read_text())
'''
for code in [launch,monitor]: ast.parse(code)
p=R/'MARSO_CURSOR_CONTINUE.ipynb'; nb=json.loads(p.read_text())
assert not any(c.get('id')=='main-launch' for c in nb['cells'])
bak=R.parent/'.hermes-backups'/('continue-before-main-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'.ipynb')
shutil.copy2(p,bak)
for key,code in [('main-launch',launch),('main-monitor',monitor)]:
    nb['cells'].append({'cell_type':'code','id':key,'metadata':{},'execution_count':None,'outputs':[],
                       'source':code.splitlines(keepends=True)})
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('APPENDED',p,'BACKUP',bak)
