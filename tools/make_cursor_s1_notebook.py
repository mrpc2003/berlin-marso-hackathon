"""Build the isolated, credential-free Cursor/Colab S1 notebook."""
from __future__ import annotations
import ast
import json
from pathlib import Path
from textwrap import dedent

SETUP = r'''
import os, sys, json, time, subprocess, hashlib, shutil
from pathlib import Path
REPO = Path('/content/berlin-marso-hackathon')
ROOT = Path('/content/drive/MyDrive/marso')
VENV = Path('/content/marso-py312')
PY = str(VENV / 'bin/python')
RUN_ID = time.strftime('%Y%m%d-%H%M%S')
ENV = dict(os.environ, DISPLAY='', PYOPENGL_PLATFORM='egl',
           HDF5_USE_FILE_LOCKING='FALSE', PYTHONUNBUFFERED='1',
           PYTHONPATH=str(REPO), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')

def run(cmd, label, cwd=REPO, extra_env=None):
    log = ROOT / 'logs' / f'{RUN_ID}_{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with log.open('w') as f:
        p = subprocess.Popen([str(x) for x in cmd], cwd=str(cwd),
                             env={**ENV, **(extra_env or {})}, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        try:
            for line in p.stdout:
                f.write(line); f.flush(); chunks.append(line)
                # Avoid flooding the notebook with progress-bar updates.
                if 'it/s' not in line and 's/it' not in line:
                    print(line, end='', flush=True)
            rc = p.wait()
        finally:
            p.stdout.close()
        if rc:
            raise RuntimeError(f'{label} failed: exit={rc}; log={log}\n' + ''.join(chunks)[-6000:])
    return ''.join(chunks)

assert (REPO / 'il/train.py').is_file(), 'Expected the existing cloned repository'
assert (ROOT / 'RUNLOG.md').is_file(), 'Mount the existing Drive before running'
for d in ('logs', 'evals', 'ckpts', 'demos/easy', 'runs'):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], 'uv', cwd='/content')
if not Path(PY).is_file():
    run([sys.executable, '-m', 'uv', 'venv', '--seed', '--python', '3.12', VENV], 'venv', cwd='/content')
uv = [sys.executable, '-m', 'uv', 'pip', 'install', '--python', PY]
run(uv + ['--index-url', 'https://download.pytorch.org/whl/cu128',
          'torch==2.11.0', 'torchvision==0.26.0'], 'torch')
run(uv + ['numpy<2', 'mani-skill==3.0.1', 'gymnasium==1.3.0', 'diffusers==0.38.0',
          'hydra-core', 'omegaconf', 'tyro', 'h5py', 'tensorboard', 'pytest'], 'dependencies')
run(uv + ['-e', str(REPO)], 'editable')
probe = """
import json, sys, torch, torchvision, mani_skill, gymnasium, diffusers, h5py, hydra, tyro
from importlib.metadata import version
assert torch.cuda.is_available()
a = torch.randn(128,128,device='cuda'); b = a @ a.T; torch.cuda.synchronize()
assert torch.isfinite(b).all()
print(json.dumps({'python': sys.version.split()[0], 'gpu':torch.cuda.get_device_name(),
 'cuda':torch.version.cuda,'kernel_smoke':True,'versions':{n:version(n) for n in
 ['torch','torchvision','numpy','mani-skill','sapien','mplib','gymnasium','diffusers']}}, indent=2))
"""
run([PY, '-c', probe], 'cuda_probe')
print('SETUP_OK', PY)
'''

DATA_PROBE = r'''
import json, os, hashlib
from pathlib import Path
import h5py, numpy as np
p = Path(os.environ['MARSO_H5'])
meta = json.loads(p.with_suffix('.json').read_text())
assert meta['env_info']['env_id'] == 'WarehouseSort-v1'
lengths, max_abs, over, count = [], 0., 0, 0
with h5py.File(p, 'r') as f:
    keys = sorted(f.keys(), key=lambda k: int(k.split('_')[-1]))
    assert len(keys) == 200, len(keys)
    for k in keys:
        t = f[k]; a=t['actions'][()]; T=len(a); lengths.append(T)
        assert a.shape == (T,4) and np.isfinite(a).all()
        for key, d in [('obs/agent/qpos',9),('obs/agent/qvel',9),('obs/extra/tcp_pose',7)]:
            x=t[key][()]; assert x.shape == (T+1,d) and np.isfinite(x).all(), (k,key,x.shape)
        assert t['obs/extra/is_grasped'].shape in [(T+1,), (T+1,1)]
        rgb=t['obs/sensor_data/scene_camera/rgb']
        assert rgb.shape == (T+1,128,128,3) and rgb.dtype == np.uint8
        # Read every frame, not just the HDF5 header.
        assert rgb[()].shape == rgb.shape
        max_abs=max(max_abs,float(np.abs(a).max()))
        over+=int((np.abs(a[:,:3])>1).sum()); count+=a[:,:3].size
summary={'h5_trajectories':len(keys),'json_episode_records':len(meta['episodes']),
 'length_min':min(lengths),'length_max':max(lengths),'length_mean':float(np.mean(lengths)),
 'max_abs_action':max_abs,'xyz_components_outside_range_fraction':over/count,
 'h5_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'json_sha256':hashlib.sha256(p.with_suffix('.json').read_bytes()).hexdigest()}
Path(os.environ['MARSO_DATA_REPORT']).write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if len(meta['episodes'])!=len(keys):
    print('NOTE: sidecar episode records are partial; env_info is present. No metadata was invented.')
'''

DATA_SETUP = r'''
name = 'trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5'
candidates = [Path('/content/easy')/name, Path('/content/demos/easy')/name,
              Path('/content')/name, REPO/'il/demos/easy'/name, ROOT/'demos/easy'/name]
sources = [p for p in candidates if p.is_file() and p.with_suffix('.json').is_file()]
assert sources, 'Upload the easy folder using Explorer > Upload to Colab first'
src = sources[0]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
for root in (REPO/'il/demos/easy', ROOT/'demos/easy'):
    root.mkdir(parents=True,exist_ok=True)
    for source in (src,src.with_suffix('.json')):
        dst=root/source.name
        if dst.exists():
            assert sha(dst)==sha(source), f'Existing data differs; not overwriting {dst}'
        else:
            shutil.copy2(source,dst)
        assert sha(dst)==sha(source)
DATA_REPORT=ROOT/'evals'/f'{RUN_ID}_data.json'
run([PY,'-c',DATA_PROBE], 'data_validation', extra_env={
    'MARSO_H5':str(REPO/'il/demos/easy'/name),'MARSO_DATA_REPORT':str(DATA_REPORT)})
print('DATA_OK', json.loads(DATA_REPORT.read_text())['h5_trajectories'])
'''

SIM_PROBE = r'''
import os, json
from pathlib import Path
import numpy as np, torch, gymnasium as gym
import warehouse_sort
from examples.scripted_policy import scripted_episode
kwargs=dict(num_envs=1,control_mode='pd_ee_delta_pos',sim_backend='gpu',
 difficulty='easy',num_parcels=2,fixed_poses=True,max_episode_steps=250)
env=gym.make('WarehouseSort-v1',obs_mode='rgb',render_mode='rgb_array',**kwargs)
try:
    env.reset(seed=42); frame=env.render()
    frame=frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    if frame.ndim==4 and frame.shape[0]==1: frame=frame[0]
    assert frame.ndim==3 and frame.shape[-1] in (3,4),frame.shape
    history=scripted_episode(env,max_steps=250,seed=42)
    sorted_count=int(history[-1][-1]['success_count'].item())
    assert sorted_count==2,(sorted_count,len(history))
finally:
    env.close()
env=gym.make('WarehouseSort-v1',obs_mode='state',**kwargs)
try:
    def displacement(ax):
        env.reset(seed=0); p0=env.unwrapped.agent.tcp_pose.p[0].clone()
        for _ in range(3): env.step(torch.tensor([[ax,0.,0.,1.]]))
        return (env.unwrapped.agent.tcp_pose.p[0]-p0).detach().cpu().numpy()
    big,one=displacement(3.66),displacement(1.)
    assert np.allclose(big,one,atol=1e-4),(big,one)
finally:
    env.close()
report={'scripted_sorted':sorted_count,'scripted_steps':len(history),'render_shape':list(frame.shape),
 'action_3_66_displacement':big.tolist(),'action_1_0_displacement':one.tolist(),
 'clipping_max_difference':float(np.max(np.abs(big-one))),'passed':True}
Path(os.environ['MARSO_SIM_REPORT']).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
'''

SIM_RUN = r'''
SIM_REPORT=ROOT/'evals'/f'{RUN_ID}_sim.json'
run([PY,'-c',SIM_PROBE], 'sim_validation',extra_env={'MARSO_SIM_REPORT':str(SIM_REPORT)})
assert json.loads(SIM_REPORT.read_text())['passed']
print('SIM_OK')
'''

SMOKE = r'''
import re, math
assert json.loads(SIM_REPORT.read_text())['passed']
smoke_exp='cursor_smoke_'+RUN_ID
speed_exp='cursor_speed_'+RUN_ID
base=[PY,'il/train.py','method=dp_rgb_easy']
smoke_output=run(base+[f'flags.exp_name={smoke_exp}','flags.total_iters=300',
 'flags.eval_freq=300','flags.num_eval_episodes=8','flags.num_demos=20',
 'flags.log_freq=50','flags.save_freq=100'],'smoke_train')
ckpt=REPO/'il/baselines/diffusion_policy/runs'/smoke_exp/'checkpoints/final.pt'
assert ckpt.is_file(),ckpt
cfg=ROOT/'evals'/f'{RUN_ID}_eval8.yaml'
cfg.write_text('eval:\n  n_episodes: 8\n  seeds: [5000,5001,5002,5003,5004,5005,5006,5007]\n')
result=ROOT/'evals'/f'{RUN_ID}_smoke.jsonl'
run([PY,'eval.py','difficulty=easy','obs_mode=rgb','policy=warehouse_sort.il_policy:load_dp_rgb',
 f'checkpoint={ckpt}',f'eval_config={cfg}','record_video=false',f'results_file={result}'],'smoke_eval')
rows=[json.loads(line) for line in result.read_text().splitlines() if line.strip()]
assert rows, 'No evaluation results'
start=time.monotonic()
speed_output=run(base+[f'flags.exp_name={speed_exp}','flags.total_iters=200',
 'flags.eval_freq=0','flags.num_demos=200','flags.save_freq=0','flags.log_freq=50'],'speed_train')
elapsed=time.monotonic()-start
# Preserve NaN/Inf in parsing so numerical failure cannot disappear.
def losses(text):
    return [float(v) for v in re.findall(r'loss[=:]\s*([-+0-9.eE]+|nan|inf)',text,re.I)]
for label,text in [('smoke',smoke_output),('speed',speed_output)]:
    vals=losses(text)
    assert vals and all(math.isfinite(x) for x in vals),(label,vals)
rates=[float(v) for v in re.findall(r'([0-9.]+)it/s',speed_output)]
summary={'run_id':RUN_ID,'smoke_exp':smoke_exp,'smoke_checkpoint':str(ckpt),
 'smoke_eval':rows,'speed_exp':speed_exp,'speed_wall_seconds':elapsed,
 'training_it_s':rates[-1] if rates else None,
 'projected_30000_minutes_including_repeated_setup_overhead':elapsed/200*30000/60,
 'smoke_last_loss':losses(smoke_output)[-1],'speed_last_loss':losses(speed_output)[-1],
 'main_training_started':False}
SUMMARY=ROOT/'evals'/f'{RUN_ID}_s1_summary.json'
SUMMARY.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('SMOKE_OK; main training has not started')
'''


def cell(kind, source, identifier):
    source = dedent(source).strip()+"\n"
    if kind == 'code': ast.parse(source)
    result={'cell_type':kind,'id':identifier,'metadata':{},'source':source.splitlines(keepends=True)}
    if kind == 'code': result.update(execution_count=None,outputs=[])
    return result


def build_notebook():
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{
        'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'}},'cells':[
        cell('markdown','# Cursor Colab S1\n기존 Colab 커널은 유지하고, Python 3.12 환경에서 설치·시뮬레이션·학습을 실행합니다.\n실패하면 다음 단계로 넘어가지 않습니다. 제출과 푸시는 포함하지 않습니다.','intro'),
        cell('code',SETUP,'setup'),
        cell('code','DATA_PROBE = '+repr(dedent(DATA_PROBE).strip())+'\n'+dedent(DATA_SETUP),'data'),
        cell('code','SIM_PROBE = '+repr(dedent(SIM_PROBE).strip())+'\n'+dedent(SIM_RUN),'sim'),
        cell('code',SMOKE,'smoke'),
        cell('markdown','## 본 학습 전 확인\n실측 속도와 스모크 평가를 확인한 뒤 본 학습을 시작합니다. 위 셀만 실행하면 본 학습은 시작되지 않습니다.','gate')]}


def validate_notebook(nb):
    assert len(nb['cells'])==6
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    for source in (DATA_PROBE,SIM_PROBE): ast.parse(dedent(source))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'MARSO_CURSOR_S1.ipynb'))
    args=parser.parse_args()
    nb=build_notebook(); validate_notebook(nb)
    Path(args.output).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(args.output)
