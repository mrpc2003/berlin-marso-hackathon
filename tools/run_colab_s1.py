"""Bounded Colab training -> validation supervisor. No submission or publishing."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2))
    tmp.replace(path)


def run_stage(cmd, log, timeout, update, env, cwd):
    with Path(log).open('a') as f:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        update(stage_pid=p.pid,stage_started=time.time(),log=str(log),command=cmd)
        try:
            rc=p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
            raise RuntimeError(f'Time limit reached; latest checkpoint retained: {log}')
    if rc: raise RuntimeError(f'Process exited {rc}; see {log}')
    update(stage_pid=None)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('plan')
    args=ap.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    repo=Path(plan['repo']); root=Path(plan['root']); py=plan['python']; exp=plan['exp']
    assert repo==Path('/content/berlin-marso-hackathon')
    assert root==Path('/content/drive/MyDrive/marso')
    assert exp.replace('_','').replace('-','').isascii() and exp.replace('_','').replace('-','').isalnum()
    assert 1<=plan['total_iters']<=30000
    runroot=root/'runs'/exp; runroot.mkdir(parents=True,exist_ok=True)
    ckdir=root/'ckpts'/exp; ckdir.mkdir(parents=True,exist_ok=True)
    lockpath=repo/'outputs'/f'{exp}.lock'; lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Same experiment is already running')
        status=runroot/'status.json'
        state={'exp':exp,'pid':os.getpid(),'started':time.time(),'total_iters':plan['total_iters'],
               'status':'starting','validation_only':True,'submission_executed':False}
        def update(**kw):
            state.update(kw); state['updated']=time.time(); save_json(status,state)
        if status.exists() and json.loads(status.read_text()).get('status')=='completed':
            raise SystemExit('Already completed; not rerunning')
        try:
            for rel,expected in plan['source_sha256'].items():
                assert sha(repo/rel)==expected, f'Source changed: {rel}'
            save_json(runroot/'plan.json',plan)
            env=dict(os.environ,DISPLAY='',PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',
                     HDF5_USE_FILE_LOCKING='FALSE',PYTHONPATH=str(repo),
                     PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            train=[py,'il/train.py','method=dp_rgb_easy',f'flags.exp_name={exp}',
                   f'flags.total_iters={plan["total_iters"]}',f'flags.ckpt_dir={ckdir}',
                   'flags.save_freq=1000','flags.eval_freq=5000','flags.num_eval_episodes=32']
            latest=ckdir/'latest.pt'
            if latest.exists():
                import torch
                ck=torch.load(latest,map_location='cpu',weights_only=False)
                assert ck['config']['total_iters']==plan['total_iters'], 'Resume schedule mismatch'
                train.append(f'flags.resume={latest}')
                del ck
            update(status='training')
            run_stage(train,runroot/'train.log',7200,update,env,repo)
            import torch
            final=ckdir/'final.pt'
            ck=torch.load(final,map_location='cpu',weights_only=False)
            assert ck['iteration']==plan['total_iters'], 'Incomplete final checkpoint'
            del ck
            best=ckdir/'best_eval_sort_accuracy.pt'
            if not best.exists(): best=final
            best_hash=sha(best)
            update(status='evaluating',checkpoint=str(best),checkpoint_sha256=best_hash)
            cfg=runroot/'eval64.yaml'
            cfg.write_text('eval:\n  n_episodes: 64\n  seeds: '+json.dumps(list(range(5000,5064)))+'\n')
            results=[]
            for horizon,steps in [(8,16),(4,16),(8,32),(4,32)]:
                label=f'eval_h{horizon}_d{steps}'
                output=runroot/f'{label}_{int(time.time())}.jsonl'
                cmd=[py,'eval.py','difficulty=easy','obs_mode=rgb',
                     'policy=warehouse_sort.il_policy:load_dp_rgb',f'checkpoint={best}',
                     f'eval_config={cfg}','record_video=false',f'results_file={output}',
                     f'+policy_kwargs.act_horizon={horizon}',f'+policy_kwargs.num_inference_steps={steps}']
                update(evaluation=label)
                run_stage(cmd,runroot/f'{label}.log',1800,update,env,repo)
                rows=[json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                assert len(rows)==1 and rows[0]['n_episodes']==64
                assert rows[0]['requested_n_episodes']==64 and rows[0]['obs_mode']=='rgb'
                assert sha(best)==best_hash, 'Checkpoint changed during evaluation'
                results.append(rows[0]); save_json(runroot/'evaluation_results.json',results)
            winner=max(results,key=lambda row:row['sort_accuracy'])
            score=winner['sort_accuracy']
            preview_cfg=runroot/'eval1.yaml'
            preview_cfg.write_text('eval:\n  n_episodes: 1\n  seeds: [5000]\n'.replace('\n', '\n'))
            opts=[f'+policy_kwargs.{k}={v}' for k,v in winner['policy_kwargs'].items()]
            preview_cmd=[py,'eval.py','difficulty=easy','obs_mode=rgb',
                'policy=warehouse_sort.il_policy:load_dp_rgb',f'checkpoint={best}',
                f'eval_config={preview_cfg}','record_video=true','video_envs=1',
                f'hydra.run.dir={runroot / "preview"}']+opts
            update(status='recording_preview')
            run_stage(preview_cmd,runroot/'preview.log',1200,update,env,repo)
            videos=sorted((runroot/'preview/videos').glob('*.mp4'))
            assert videos, 'Preview did not produce an MP4'
            diag=runroot/'diagnostics.json'
            diag_cmd=[py,'tools/rollout_logger.py','difficulty=easy','obs_mode=rgb',
                'policy=warehouse_sort.il_policy:load_dp_rgb',f'checkpoint={best}',
                f'eval_config={cfg}','num_envs=8','+log_episodes=8',f'+log_out={diag}']+opts
            update(status='diagnostics')
            run_stage(diag_cmd,runroot/'diagnostics.log',1200,update,env,repo)
            assert diag.is_file()
            gate='DP main line' if score>=0.4 else ('ACT hedge next' if score>=0.3 else 'diagnose + retrain before Gate 1b')
            report={'status':'completed','exp':exp,'iterations':plan['total_iters'],
                    'checkpoint':str(best),'checkpoint_sha256':best_hash,'best_eval':winner,
                    'eval_runs':results,'gate_1':gate,'validation_only':True,
                    'videos':[{'path':str(v),'sha256':sha(v)} for v in videos],
                    'diagnostics':str(diag),
                    'submission_executed':False,'git_push_executed':False}
            save_json(runroot/'summary.json',report)
            update(status='completed',best_sort_accuracy=score,gate_1=gate,
                   summary=str(runroot/'summary.json'),finished=time.time())
        except Exception as exc:
            update(status='failed',error=f'{type(exc).__name__}: {exc}',finished=time.time())
            raise


if __name__=='__main__': main()
