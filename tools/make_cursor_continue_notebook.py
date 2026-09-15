"""Resume the existing S1 kernel without repeating the completed smoke training."""
import json
from pathlib import Path

code = r'''
import json, math, time, re
from pathlib import Path
assert Path(PY).is_file() and ckpt.is_file()
assert json.loads(SIM_REPORT.read_text())['passed']
# Non-empty kwargs avoids the confirmed OmegaConf empty-dict bug, with the same policy defaults.
retest=ROOT/'evals'/f'{RUN_ID}_smoke_retest.jsonl'
run([PY,'eval.py','difficulty=easy','obs_mode=rgb','policy=warehouse_sort.il_policy:load_dp_rgb',
 f'checkpoint={ckpt}',f'eval_config={cfg}','record_video=false',f'results_file={retest}',
 '+policy_kwargs.act_horizon=8','+policy_kwargs.num_inference_steps=16'],'smoke_eval_retest')
rows=[json.loads(x) for x in retest.read_text().splitlines() if x.strip()]
assert rows[-1]['n_episodes']==8 and rows[-1]['level']=='easy' and rows[-1]['obs_mode']=='rgb'
start=time.monotonic()
speed_output=run(base+[f'flags.exp_name={speed_exp}','flags.total_iters=200',
 'flags.eval_freq=0','flags.num_demos=200','flags.save_freq=0','flags.log_freq=50'],'speed_train')
elapsed=time.monotonic()-start

def losses(text):
    return [float(v) for v in re.findall(r'loss[=:]\s*([-+0-9.eE]+|nan|inf)',text,re.I)]
for label,text in [('smoke',smoke_output),('speed',speed_output)]:
    vals=losses(text)
    assert vals and all(math.isfinite(x) for x in vals),(label,vals)
rates=[float(v) for v in re.findall(r'([0-9.]+)\s*it/s',speed_output)]
summary={'run_id':RUN_ID,'smoke_exp':smoke_exp,'smoke_checkpoint':str(ckpt),
 'smoke_eval':rows[-1],'speed_exp':speed_exp,'speed_wall_seconds':elapsed,
 'training_it_s':rates[-1] if rates else None,
 'projected_30000_minutes_including_repeated_setup_overhead':elapsed/200*30000/60,
 'smoke_last_loss':losses(smoke_output)[-1],'speed_last_loss':losses(speed_output)[-1],
 'main_training_started':False}
SUMMARY=ROOT/'evals'/f'{RUN_ID}_s1_summary.json'
SUMMARY.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('SMOKE_AND_SPEED_OK')
'''
if __name__=='__main__':
    import ast
    ast.parse(code)
    root=Path(__file__).resolve().parents[1]
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','language':'python','display_name':'Python 3 (ipykernel)'}},'cells':[
        {'cell_type':'code','id':'recover-smoke','metadata':{},'execution_count':None,'outputs':[],'source':code.strip().splitlines(keepends=True)}]}
    (root/'MARSO_CURSOR_CONTINUE.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
