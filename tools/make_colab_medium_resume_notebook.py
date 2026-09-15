"""Build a separate recovery notebook; never edit original executed notebooks."""
from __future__ import annotations
import ast
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from textwrap import dedent, indent

import tools.make_colab_medium_notebook as base
import tools.run_colab_medium_resume as supervisor

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = supervisor.EXPECTED_RESUME_SHA256
PRIOR_BEST_SHA = supervisor.EXPECTED_PRIOR_BEST_SHA256
PRIOR_EXP = supervisor.EXPECTED_PRIOR_EXP
DRIVE_INPUT_SOURCE = supervisor.EXPECTED_DRIVE_INPUT_SOURCE
OUTPUT = REPO/'MARSO_COLAB_MEDIUM_RESUME_29000.ipynb'
FILES = base.EMBEDDED_FILES + [
    'il/baselines/diffusion_policy/train_rgbd.py',
    'tools/run_colab_medium_resume.py',
]

SETUP = base.SETUP.replace(
    "for d in ('logs','runs/medium','ckpts/medium','evals/medium','demos/medium','locks'): (ROOT/d).mkdir(parents=True, exist_ok=True)",
    "assert os.path.ismount('/content/drive'), 'Drive must be mounted, not an ordinary directory'\n"
    "assert (ROOT/'demos/medium/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5').is_file()\n"
    "LOCAL_SETUP = Path('/content/marso-recovery-setup')/RUN_ID\nLOCAL_SETUP.mkdir(parents=True, exist_ok=False)"
).replace("log = ROOT/'logs'/f'{RUN_ID}_{label}.log'", "log = LOCAL_SETUP/f'{label}.log'")
SETUP = SETUP.replace("saved=ROOT/'backups'/('medium-src-'+RUN_ID)/rel", "saved=LOCAL_SETUP/'source-backups'/rel")
SETUP = SETUP.replace("if 'it/s' not in line and 's/it' not in line: print(line, end='', flush=True)", "print(line, end='', flush=True)")
# Keep the base setup unchanged; probe the complete existing environment before installing.
_install_start = SETUP.index("run([sys.executable,'-m','pip','install','-q','uv'], 'uv')")
_probe_start = SETUP.index('\nprobe = ', _install_start)
_install = SETUP[_install_start:_probe_start]
RUNTIME_SETUP = SETUP[_probe_start:].replace(
    'from importlib.metadata import version',
    "import numpy, omegaconf, tensorboard, pytest, mplib, warehouse_sort\n"
    "from diffusers.training_utils import EMAModel\n"
    "from importlib.metadata import version\n"
    "for name, expected in {'torch':'2.11.0','torchvision':'0.26.0','mani-skill':'3.0.1','gymnasium':'1.3.0','diffusers':'0.38.0','mplib':'0.1.1'}.items():\n"
    "    assert version(name).split('+')[0] == expected, (name, version(name))\n"
    "assert int(version('numpy').split('.')[0]) < 2"
).replace("run([PY,'-c',probe], 'cuda_probe')", """reuse_valid = False
if Path(PY).is_file():
    try:
        run([PY,'-c',probe], 'reuse_probe')
        reuse_valid = True
    except RuntimeError as exc:
        print('Existing Python 3.12 environment needs setup:', exc)
if reuse_valid:
    print('Reusing validated Python 3.12 environment:', PY)
else:
""" + indent(_install, '    ') + "\n    run([PY,'-c',probe], 'cuda_probe')")
SETUP = SETUP[:_install_start] + RUNTIME_SETUP
SETUP += "\nprint('Recovery setup complete. No training started.')\n"

STAGE = base.STAGE.replace("DATA_REPORT=ROOT/'evals/medium'/f'{RUN_ID}_data.json'", "DATA_REPORT=LOCAL_SETUP/'data.json'") + r'''
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.run_colab_medium_resume import atomic_copy_verified, ensure_remote_ready, load_plan
ensure_remote_ready(ROOT)
OLD_EXP = PRIOR_EXP
EXP = f'medium_resume_29000_{RUN_ID}'
assert EXP != OLD_EXP
LOCAL_ROOT = Path('/content/marso-resume')/EXP
assert not LOCAL_ROOT.exists(), 'Use a new recovery run name'
assert not (ROOT/'recoveries/medium'/EXP).exists(), 'Do not reuse a Drive recovery run'
INPUTS = LOCAL_ROOT/'inputs'
INPUTS.mkdir(parents=True, exist_ok=True)
RESUME_CHECKPOINT = INPUTS/'latest.pt'
PRIOR_BEST_CHECKPOINT = INPUTS/'best_eval_sort_accuracy.pt'
old_ckpts = ROOT/'recoveries/medium'/OLD_EXP/'ckpts'
assert DRIVE_INPUT_SOURCE == 'gdrive:marso/' + old_ckpts.relative_to(ROOT).as_posix()
assert sha(old_ckpts/'latest.pt') == CHECKPOINT_SHA, 'Prior latest changed; stop for verification'
assert sha(old_ckpts/'best_eval_sort_accuracy.pt') == PRIOR_BEST_SHA, 'Prior best changed; stop for verification'
for filename, expected_sha in (('latest.pt',CHECKPOINT_SHA),('best_eval_sort_accuracy.pt',PRIOR_BEST_SHA)):
    source = old_ckpts/filename
    target = INPUTS/filename
    if target.exists():
        assert sha(target) == expected_sha == sha(source), 'Do not overwrite a different recovery input'
    else:
        atomic_copy_verified(source, target, guard=lambda: ensure_remote_ready(ROOT))
    assert sha(target) == expected_sha == sha(source)
assert sha(RESUME_CHECKPOINT) == CHECKPOINT_SHA
assert sha(PRIOR_BEST_CHECKPOINT) == PRIOR_BEST_SHA
run([PY,'-c',CHECKPOINT_PROBE,str(RESUME_CHECKPOINT),str(PRIOR_BEST_CHECKPOINT)], 'checkpoint_validation')
source_sha256 = {rel:sha(REPO/rel) for rel in EMBEDDED_SOURCES}
tracked = subprocess.check_output(['git','ls-files','--','warehouse_sort','conf','il/conf','il/baselines/diffusion_policy','il/train.py'],cwd=REPO,text=True).splitlines()
source_sha256.update({rel:sha(REPO/rel) for rel in tracked if rel.endswith(('.py','.yaml'))})
PLAN = LOCAL_SETUP/'recovery-plan.json'
plan = {'repo':str(REPO),'python':PY,'exp':EXP,'local_root':str(LOCAL_ROOT),'drive_root':str(ROOT),
        'resume_checkpoint':str(RESUME_CHECKPOINT),'resume_sha256':CHECKPOINT_SHA,
        'prior_best_checkpoint':str(PRIOR_BEST_CHECKPOINT),'prior_best_sha256':PRIOR_BEST_SHA,
        'source_sha256':source_sha256,'level':'medium','method':'dp_rgb_medium',
        'total_iters':30000,'max_episode_steps':500,'prior_exp':OLD_EXP,'drive_input_source':DRIVE_INPUT_SOURCE}
PLAN.write_text(json.dumps(plan,indent=2))
load_plan(PLAN)
print('RESUME_INPUTS_VERIFIED',json.dumps({'source_iteration':29000,'target_iteration':30000,'exp':EXP,'plan':str(PLAN),'checkpoint_sha256':CHECKPOINT_SHA,'prior_exp':OLD_EXP,'drive_input_source':DRIVE_INPUT_SOURCE},indent=2))
'''

CHECKPOINT_PROBE = r'''
import sys,torch,json
from tools.run_colab_medium_resume import (
    EXPECTED_RESUME_SHA256, EXPECTED_PRIOR_BEST_SHA256, sha,
    validate_medium_checkpoint, validate_prior_best_checkpoint,
)
assert sha(sys.argv[1]) == EXPECTED_RESUME_SHA256
assert sha(sys.argv[2]) == EXPECTED_PRIOR_BEST_SHA256
resume=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
best=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert resume['iteration']==29000 and best['iteration']==25000
validate_medium_checkpoint(resume, expected_iteration=29000)
validate_prior_best_checkpoint(best)
assert set(resume)>={'agent','ema_agent','optimizer','lr_scheduler','ema_state','scaler','rng','config','args','best_eval_metrics','eval_history'}
for ck in (resume,best):
    cfg=ck['config']
    for k,v in {'total_iters':30000,'batch_size':128,'lr':1e-4,'obs_horizon':2,'act_horizon':8,'pred_horizon':16,'num_diffusion_iters':100,'visual_encoder':'resnet18','max_episode_steps':500,'clip_actions':True,'image_aug_pad':4}.items():
        assert cfg[k]==v,(k,cfg[k],v)
    assert all('/medium/' in p for p in cfg['demo_paths'])
    for key in ('agent','ema_agent'):
        assert ck[key] and all(torch.isfinite(t).all().item() for t in ck[key].values() if torch.is_tensor(t)), key
assert best['best_eval_metrics']['sort_accuracy']==1.0
assert resume['optimizer']['state'] and resume['rng']['torch'].device.type=='cpu'
print(json.dumps({'resume_iteration':resume['iteration'],'prior_best_iteration':best['iteration'],'best_sort_accuracy':best['best_eval_metrics']['sort_accuracy'],'optimizer_state_entries':len(resume['optimizer']['state']),'lr':resume['optimizer']['param_groups'][0]['lr'],'scheduler_step':resume['lr_scheduler'].get('last_epoch'),'finite_weights':True},indent=2))
'''

SIM = base.SIM.replace("SIM_REPORT=ROOT/'evals/medium'/f'{RUN_ID}_sim.json'", "SIM_REPORT=LOCAL_SETUP/'sim.json'")

LAUNCH = r'''
#@title Resume medium training and final evaluation
RUN_RESUME = False #@param {type:"boolean"}
if not RUN_RESUME:
    print('Not started. Enable only this cell after input/CUDA/reference checks.')
else:
    assert json.loads(SIM_REPORT.read_text())['passed']
    assert sha(RESUME_CHECKPOINT)==CHECKPOINT_SHA
    assert sha(PRIOR_BEST_CHECKPOINT)==PRIOR_BEST_SHA
    print('Starting from checkpoint iteration 29000; target remains 30000. This cell stays running until the pipeline exits.',flush=True)
    print('Prior experiment is read-only. Local outputs:',LOCAL_ROOT,flush=True)
    result = run([PY,'-u',str(REPO/'tools/run_colab_medium_resume.py'),str(PLAN)],'recovery_supervisor',cwd=REPO)
    print(result[-4000:])
'''

MONITOR = r'''
import json,subprocess
from pathlib import Path
local_base=Path('/content/marso-resume')
drive_base=Path('/content/drive/MyDrive/marso/recoveries/medium')
roots=list(local_base.glob('medium_resume_*')) if local_base.is_dir() else []
if not roots and drive_base.is_dir():
    roots=list(drive_base.glob('medium_resume_*'))
if not roots:
    print('No recovery output found; do not relaunch blindly.')
else:
    root=max(roots,key=lambda p:p.stat().st_mtime)
    print('Observed recovery',root)
    for rel in ('run/status.json','run/summary.json','run/train.log','status.json','summary.json'):
        p=root/rel
        if p.is_file():
            with p.open('rb') as f:
                if p.suffix!='.json': f.seek(max(0,p.stat().st_size-3500))
                text=f.read().decode(errors='replace')
            print(rel,text)
'''


def build_notebook():
    embedded={}
    for rel in FILES:
        data=(REPO/rel).read_bytes()
        old=subprocess.run(['git','show',f'6a7346f:{rel}'],cwd=REPO,capture_output=True)
        embedded[rel]={'base_sha256':hashlib.sha256(old.stdout).hexdigest() if old.returncode==0 else None,
                       'sha256':hashlib.sha256(data).hexdigest(),'content_b64':base64.b64encode(data).decode()}
    sources=[
        ('resume-drive',base.CELL0),
        ('resume-setup','EMBEDDED_SOURCES = '+repr(embedded)+'\n'+SETUP),
        ('resume-inputs','CHECKPOINT_SHA = '+repr(CHECKPOINT_SHA)+'\nPRIOR_BEST_SHA = '+repr(PRIOR_BEST_SHA)+'\nPRIOR_EXP = '+repr(PRIOR_EXP)+'\nDRIVE_INPUT_SOURCE = '+repr(DRIVE_INPUT_SOURCE)+'\nDATA_PROBE = '+repr(dedent(base.DATA_PROBE))+'\nCHECKPOINT_PROBE = '+repr(dedent(CHECKPOINT_PROBE))+'\n'+STAGE),
        ('resume-reference','SIM_PROBE = '+repr(dedent(base.SIM_PROBE))+'\n'+SIM),
        ('resume-launch',LAUNCH),('resume-monitor',MONITOR),
    ]
    for probe in [CHECKPOINT_PROBE,base.DATA_PROBE,base.SIM_PROBE]: ast.parse(dedent(probe))
    cells=[base.cell(source,cid) for cid,source in sources]
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    import nbformat
    nbformat.validate(nb)
    return nb

if __name__=='__main__':
    out=OUTPUT
    out.write_text(json.dumps(build_notebook(),ensure_ascii=False,indent=1)+'\n')
    print(out)
