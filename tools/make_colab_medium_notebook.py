"""Build the medium-only Colab notebook from executable Python cell sources."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "MARSO_COLAB_MEDIUM.ipynb"
BRANCH = "feat/colab-t4-rgb-dp"
BASE_COMMIT = "6a7346f"
EMBEDDED_FILES = [
    "eval.py",
    "tools/rollout_logger.py",
    "il/baselines/diffusion_policy/diffusion_policy/evaluate.py",
    "tools/run_colab_medium.py",
]

CELL0 = "from google.colab import drive\ndrive.mount('/content/drive')\n"

SETUP = r'''
import os, sys, json, time, subprocess, hashlib, textwrap, base64
from pathlib import Path
REPO = Path('/content/berlin-marso-hackathon')
ROOT = Path('/content/drive/MyDrive/marso')
VENV = Path('/content/marso-py312')
PY = str(VENV / 'bin/python')
RUN_ID = time.strftime('%Y%m%d-%H%M%S')
ENV = dict(os.environ, DISPLAY='', PYOPENGL_PLATFORM='egl', HDF5_USE_FILE_LOCKING='FALSE',
           PYTHONUNBUFFERED='1', PYTHONPATH=str(REPO), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive first'
for d in ('logs','runs/medium','ckpts/medium','evals/medium','demos/medium','locks'): (ROOT/d).mkdir(parents=True, exist_ok=True)
def run(cmd, label, cwd='/content', extra_env=None):
    log = ROOT/'logs'/f'{RUN_ID}_{label}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with log.open('w') as f:
        p = subprocess.Popen([str(x) for x in cmd], cwd=str(cwd), env={**ENV, **(extra_env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush(); out.append(line)
            if 'it/s' not in line and 's/it' not in line: print(line, end='', flush=True)
        rc = p.wait()
    if rc: raise RuntimeError(f'{label} failed exit={rc}; log={log}\n' + ''.join(out)[-6000:])
    return ''.join(out)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
print(run(['nvidia-smi'], 'nvidia_smi'))
if not REPO.exists():
    run(['git','clone','--branch','feat/colab-t4-rgb-dp','https://github.com/mrpc2003/berlin-marso-hackathon.git',str(REPO)], 'clone')
assert run(['git','rev-parse','HEAD'], 'base_commit', cwd=REPO).strip().startswith('6a7346f'), 'Unexpected checkout: refusing to reset or update it'
for rel,item in EMBEDDED_SOURCES.items():
    p=REPO/rel; actual=sha(p) if p.exists() else None
    assert actual in (item['base_sha256'],item['sha256']), f'Unexpected existing source: {rel}'
for rel, item in EMBEDDED_SOURCES.items():
    path = REPO / rel; path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) or sha(path) != item['sha256']:
        if path.exists():
            import shutil
            saved=ROOT/'backups'/('medium-src-'+RUN_ID)/rel
            saved.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,saved)
        path.write_bytes(base64.b64decode(item['content_b64']))
    assert sha(path) == item['sha256'], rel
run([sys.executable,'-m','pip','install','-q','uv'], 'uv')
run([sys.executable,'-m','uv','python','install','3.12.3'], 'uv_python')
if not Path(PY).is_file():
    run([sys.executable,'-m','uv','venv','--seed','--python','3.12.3',str(VENV)], 'venv')
uv=[sys.executable,'-m','uv','pip','install','--python',PY]
run(uv+['--index-url','https://download.pytorch.org/whl/cu128','torch==2.11.0','torchvision==0.26.0'], 'torch')
run(uv+['numpy<2','mani-skill==3.0.1','gymnasium==1.3.0','diffusers==0.38.0','hydra-core','omegaconf','tyro','h5py','tensorboard','pytest'], 'deps')
run(uv+['-e',str(REPO)], 'editable')
probe = """
import json, sys, torch, torchvision, mani_skill, gymnasium, diffusers, h5py, hydra, tyro
from importlib.metadata import version
assert sys.version.split()[0].startswith('3.12.3')
assert torch.cuda.is_available()
x=torch.randn(256,256,device='cuda'); y=x@x.T; torch.cuda.synchronize(); assert torch.isfinite(y).all()
print(json.dumps({'python':sys.version.split()[0],'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
'versions':{n:version(n) for n in ['torch','torchvision','numpy','mani-skill','gymnasium','diffusers']}}, indent=2))
"""
run([PY,'-c',probe], 'cuda_probe')
print('SETUP_OK', PY, 'run_id', RUN_ID)
'''

DATA_PROBE = r'''
import json, os, hashlib
from pathlib import Path
import h5py, numpy as np
h5 = Path(os.environ['MARSO_H5'])
meta = json.loads(h5.with_suffix('.json').read_text())
assert meta['env_info']['env_id']=='WarehouseSort-v1'
assert meta['env_info']['env_kwargs']['num_parcels']==4 and meta['env_info']['env_kwargs']['fixed_poses'] is False
lengths = []
with h5py.File(h5, 'r') as f:
    keys = sorted(f.keys(), key=lambda k: int(k.split('_')[-1]))
    assert len(keys) == 200, len(keys)
    for k in keys:
        g=f[k]; a=g['actions'][()]; T=a.shape[0]; lengths.append(int(T))
        assert a.shape == (T,4) and np.isfinite(a).all(), (k,a.shape)
        for name, dim in [('obs/agent/qpos',9),('obs/agent/qvel',9),('obs/extra/tcp_pose',7)]:
            x=g[name][()]; assert x.shape == (T+1,dim) and np.isfinite(x).all(), (k,name,x.shape)
        grasp=g['obs/extra/is_grasped']; assert grasp.shape in ((T+1,), (T+1,1)), (k,grasp.shape)
        rgb=g['obs/sensor_data/scene_camera/rgb']; assert rgb.shape == (T+1,128,128,3) and rgb.dtype == np.uint8, (k,rgb.shape,rgb.dtype)
        assert rgb[()].shape == rgb.shape
episodes = meta.get('episodes', [])
summary = {'h5_trajectories':len(keys),'json_episode_records':len(episodes),'metadata_episodecount_mismatch':len(episodes)!=len(keys),
 'length_min':min(lengths),'length_max':max(lengths),'lengths':lengths,
 'h5_sha256':hashlib.sha256(h5.read_bytes()).hexdigest(),'json_sha256':hashlib.sha256(h5.with_suffix('.json').read_bytes()).hexdigest()}
Path(os.environ['MARSO_DATA_REPORT']).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if summary['metadata_episodecount_mismatch']: print('WARNING metadata episode count mismatch; no metadata invented')
'''

STAGE = r'''
name='trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5'
src=ROOT/'demos/medium'/name
assert src.is_file() and src.with_suffix('.json').is_file(), 'Upload Drive MyDrive/marso/demos/medium h5+json via the official Drive UI'
assert sha(src)=='a9572cd0c2f8c95be49f4af3fdfec16827477e392f42c9a1d92a7350c3edeeb0'
assert sha(src.with_suffix('.json'))=='267243b11efb7b354f2c5c8a404d98aff1afb6cf98bef59a0b01aab8d7960b11'
dst_dir=REPO/'il/demos/medium'; dst_dir.mkdir(parents=True, exist_ok=True)
for source in (src, src.with_suffix('.json')):
    dst=dst_dir/source.name
    if dst.exists(): assert sha(dst)==sha(source), f'Existing runtime copy differs: {dst}'
    else: import shutil; shutil.copy2(source, dst)
    assert sha(dst)==sha(source)
DATA_REPORT=ROOT/'evals/medium'/f'{RUN_ID}_data.json'
run([PY,'-c',DATA_PROBE], 'stage_medium', extra_env={'MARSO_H5':str(dst_dir/name),'MARSO_DATA_REPORT':str(DATA_REPORT)})
print('DATA_OK', json.loads(DATA_REPORT.read_text())['h5_trajectories'])
'''

SIM_PROBE = r'''
import json, os
from pathlib import Path
import gymnasium as gym, numpy as np, torch
import warehouse_sort
from examples.scripted_policy import DIFFICULTY_KWARGS, scripted_episode
kwargs=dict(DIFFICULTY_KWARGS['medium'])
assert kwargs['num_parcels']==4 and kwargs['fixed_poses'] is False
assert kwargs['randomization']['parcel_pose']['xy_jitter']==[-0.015,0.015]
env=gym.make('WarehouseSort-v1', num_envs=1, obs_mode='rgb', control_mode='pd_ee_delta_pos',
             sim_backend='gpu', render_mode='rgb_array', max_episode_steps=500, **kwargs)
try:
    env.reset(seed=5000); frame=env.render()
    frame=frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    if frame.ndim==4: frame=frame[0]
    assert frame.ndim==3 and frame.shape[-1] in (3,4), frame.shape
    hist=scripted_episode(env, max_steps=500, seed=5000)
    sorted_count=int(hist[-1][-1]['success_count'].item())
    assert sorted_count==4, (sorted_count, len(hist))
finally:
    env.close()
report={'difficulty':'medium','num_parcels':4,'max_episode_steps':500,'render_shape':list(frame.shape),'scripted_sorted':sorted_count,'passed':True}
Path(os.environ['MARSO_SIM_REPORT']).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
'''

SIM = r'''
SIM_REPORT=ROOT/'evals/medium'/f'{RUN_ID}_sim.json'
run([PY,'-c',SIM_PROBE], 'medium_reference', extra_env={'MARSO_SIM_REPORT':str(SIM_REPORT)})
assert json.loads(SIM_REPORT.read_text())['passed']
print('SIM_OK')
'''

SMOKE = r'''
import math, re
assert json.loads(DATA_REPORT.read_text())['h5_trajectories']==200
assert json.loads(SIM_REPORT.read_text())['scripted_sorted']==4
def losses(text):
    vals=[float(v) for v in re.findall(r'loss[=:]\s*([-+0-9.eE]+|nan|inf)', text, re.I)]
    assert vals and all(math.isfinite(v) for v in vals), vals
    return vals
smoke_exp=f'medium_smoke_{RUN_ID}'
speed_exp=f'medium_speed_{RUN_ID}'
base=[PY,'il/train.py','method=dp_rgb_medium']
smoke=run(base+[f'flags.exp_name={smoke_exp}','flags.total_iters=300','flags.eval_freq=300','flags.num_eval_episodes=8',
 'flags.num_demos=20','flags.save_freq=100','flags.log_freq=50'], 'smoke300', cwd=REPO)
losses(smoke)
ckpt=REPO/'il/baselines/diffusion_policy/runs'/smoke_exp/'checkpoints/final.pt'
assert ckpt.is_file(), ckpt
cfg=ROOT/'evals/medium'/f'{RUN_ID}_eval8.yaml'; cfg.write_text('eval:\n  n_episodes: 8\n  seeds: [5000,5001,5002,5003,5004,5005,5006,5007]\n')
result=ROOT/'evals/medium'/f'{RUN_ID}_smoke_eval.jsonl'
run([PY,'eval.py','difficulty=medium','obs_mode=rgb','policy=warehouse_sort.il_policy:load_dp_rgb',f'checkpoint={ckpt}',f'eval_config={cfg}',
 'record_video=false',f'results_file={result}'], 'smoke_eval8', cwd=REPO)
rows=[json.loads(x) for x in result.read_text().splitlines() if x.strip()]
assert len(rows)==1 and rows[0]['n_episodes']==8 and rows[0]['requested_n_episodes']==8 and rows[0]['level']=='medium' and rows[0]['max_episode_steps']==500
t0=time.monotonic()
speed=run(base+[f'flags.exp_name={speed_exp}','flags.total_iters=200','flags.eval_freq=0','flags.num_demos=200','flags.save_freq=0','flags.log_freq=50'],
          'speed200_full200', cwd=REPO)
elapsed=time.monotonic()-t0; losses(speed)
rates=[float(x) for x in re.findall(r'([0-9.]+)it/s', speed)]
summary={'run_id':RUN_ID,'smoke_exp':smoke_exp,'speed_exp':speed_exp,'speed_wall_seconds':elapsed,
 'training_it_s':rates[-1] if rates else None,'estimated_30000_minutes':elapsed/200*30000/60,'main_training_started':False}
SMOKE_SUMMARY=ROOT/'evals/medium'/f'{RUN_ID}_smoke_summary.json'; SMOKE_SUMMARY.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('SMOKE_OK main training has not started')
'''

LAUNCH = r'''
#@title 5. Medium main training
import subprocess, os, json, time
RUN_MAIN = False #@param {type:"boolean"}
EXP = f'medium_{RUN_ID}'
PLAN = ROOT/'runs/medium'/EXP/'plan.json'
estimate = json.loads(SMOKE_SUMMARY.read_text()).get('estimated_30000_minutes')
print(f'Main medium run is default-off. Estimated 30000-iter train time: {estimate:.1f} minutes before eval/video/diagnostics.')
print('Set RUN_MAIN = True in this cell only after checking the estimate and smoke outputs.')
if RUN_MAIN:
    tracked=subprocess.check_output(['git','ls-files','--','warehouse_sort','conf','il/conf','il/baselines/diffusion_policy','eval.py','tools/rollout_logger.py'],cwd=REPO,text=True).splitlines()
    source_sha256={rel:sha(REPO/rel) for rel in sorted(set([x for x in tracked if x.endswith(('.py','.yaml'))]+list(EMBEDDED_SOURCES)))}
    plan={'repo':str(REPO),'root':str(ROOT),'python':PY,'exp':EXP,'level':'medium','method':'dp_rgb_medium',
          'total_iters':30000,'max_episode_steps':500,'source_sha256':source_sha256}
    PLAN.parent.mkdir(parents=True, exist_ok=True); PLAN.write_text(json.dumps(plan, indent=2))
    log=PLAN.parent/'supervisor.log'
    with log.open('a') as f:
        child=subprocess.Popen([PY,str(REPO/'tools/run_colab_medium.py'),str(PLAN)],cwd=REPO,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1)
    assert child.poll() is None, log.read_text()[-6000:]
    state=json.loads((PLAN.parent/'status.json').read_text())
    assert state['pid']==child.pid and state['status']=='training',state
    print('MEDIUM_TRAINING_STARTED',json.dumps(state,indent=2))
else:
    print('RUN_MAIN is False; launch skipped.')
'''

MONITOR = r'''
import json, subprocess
from pathlib import Path
ROOT=Path('/content/drive/MyDrive/marso')
EXP=''  # set to e.g. medium_20260914-153000, or leave blank for latest medium run
base=ROOT/'runs/medium'
runs=sorted([p for p in base.glob('medium_*') if p.is_dir()], key=lambda p:p.stat().st_mtime)
run=runs[-1] if not EXP and runs else base/EXP
print('monitor_run', run)
try:
    print(subprocess.run(['nvidia-smi'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10).stdout)
except Exception as exc:
    print('nvidia-smi unavailable:', type(exc).__name__, exc)
for name in ('status.json','summary.json','supervisor.log','train.log'):
    p=run/name
    if p.is_file():
        print('\\n==', p, '==')
        text=p.read_text(errors='replace')
        print(text[-4000:] if p.suffix!='.json' else text)
status=run/'status.json'
if status.is_file():
    s=json.loads(status.read_text())
    ck=s.get('checkpoint')
    py=Path('/content/marso-py312/bin/python')
    if ck and py.is_file() and Path(ck).is_file():
        code="import sys, torch; ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print({'iteration':ck.get('iteration'), 'has_config':'config' in ck})"
        print(subprocess.run([str(py),'-c',code,ck], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120).stdout)
'''


def _sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def embedded_sources():
    items = {}
    for rel in EMBEDDED_FILES:
        data = (REPO_ROOT / rel).read_bytes()
        base=subprocess.run(["git","show",f"{BASE_COMMIT}:{rel}"],cwd=REPO_ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        items[rel] = {"base_sha256":hashlib.sha256(base.stdout).hexdigest() if base.returncode==0 else None, "sha256": hashlib.sha256(data).hexdigest(), "content_b64": base64.b64encode(data).decode("ascii")}
    return items


def cell(source, cid):
    source = dedent(source).strip() + "\n"
    ast.parse(source)
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(True)}


def build_notebook():
    embedded = "EMBEDDED_SOURCES = " + repr(embedded_sources()) + "\n"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [cell(CELL0, "drive-mount"), cell(embedded + SETUP, "setup-py312"), cell("DATA_PROBE = " + repr(dedent(DATA_PROBE).strip()) + "\n" + STAGE, "stage-medium"),
                      cell("SIM_PROBE = " + repr(dedent(SIM_PROBE).strip()) + "\n" + SIM, "reference-medium"), cell(SMOKE, "smoke-medium"),
                      cell(LAUNCH, "launch-medium"), cell(MONITOR, "monitor-medium")]}


def validate_notebook(nb):
    assert [c["id"] for c in nb["cells"]] == ["drive-mount", "setup-py312", "stage-medium", "reference-medium", "smoke-medium", "launch-medium", "monitor-medium"]
    for c in nb["cells"]:
        ast.parse("".join(c["source"]))
    for src in (DATA_PROBE, SIM_PROBE):
        ast.parse(dedent(src))
    try:
        import nbformat
        nbformat.validate(nb)
    except ModuleNotFoundError:
        assert nb["nbformat"] == 4 and all(c["cell_type"] == "code" for c in nb["cells"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    nb = build_notebook()
    validate_notebook(nb)
    Path(args.output).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(args.output)
