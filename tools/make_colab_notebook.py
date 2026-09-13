"""Generate MARSO_COLAB.ipynb (kept as a script so the notebook is reproducible and diff-able)."""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# WarehouseSort on Colab T4 — RGB Diffusion Policy (session notebook)

Fork workflow for the **Marso Hack Berlin 2026** challenge, tuned for the **free Colab T4** (≤3 h sessions,
12.7 GB RAM). Everything that must survive a disconnect lives on Google Drive (`MyDrive/marso/`);
code lives in git.

**Before running:** *Runtime → Change runtime type → T4 GPU*. Add these **Colab Secrets** (key icon, left bar):
`KAGGLE_API_TOKEN` (kaggle.com → Settings → API → Create New Token; the `KGAT_…` string — or legacy
`KAGGLE_USERNAME` + `KAGGLE_KEY`) and, for pushing checkpoints, `GH_TOKEN` (GitHub PAT with `repo`).

Run the cells top to bottom. Section 6 launches training in the background (`nohup`), so a UI
disconnect does not kill it; a recycled VM is handled by `--resume`.
""")

code(r"""
#@title 0. Session parameters
REPO_URL   = "https://github.com/mrpc2003/berlin-marso-hackathon.git"  #@param {type:"string"}
BRANCH     = "feat/colab-t4-rgb-dp"                                    #@param {type:"string"}
ROOT       = "/content/drive/MyDrive/marso"                            #@param {type:"string"}
LEVEL      = "easy"        #@param ["easy", "medium", "hard"]
EXP        = "rgb_dp_easy_v1"   #@param {type:"string"}
TOTAL_ITERS = 30000        #@param {type:"integer"}
EXTRA_FLAGS = ""           #@param {type:"string"}
# e.g. EXTRA_FLAGS = "flags.act_horizon=4 flags.proprio_noise_std=0.01 demo_dir=[hard,medium,easy]"
import os, time, subprocess, json, glob, shutil
REPO_DIR = "/content/berlin-marso-hackathon"
print(f"level={LEVEL} exp={EXP} iters={TOTAL_ITERS}")
""")

code(r"""
#@title 1. GPU check + Drive mount + session log
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
for d in ("demos", "ckpts", "runs", "evals", "logs", "videos"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
with open(f"{ROOT}/RUNLOG.md", "a") as f:
    f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} session start — level={LEVEL} exp={EXP} iters={TOTAL_ITERS} {EXTRA_FLAGS}\n")
print("Drive ready:", os.listdir(ROOT))
""")

code(r"""
#@title 2. Clone the fork + install (≈2-3 min)
if os.path.exists(REPO_DIR):
    !cd {REPO_DIR} && git fetch --quiet origin && git checkout -q {BRANCH} && git pull -q --ff-only origin {BRANCH}
else:
    !git clone --quiet --branch {BRANCH} {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
!git log --oneline -1
!pip install -q mani-skill==3.0.1 diffusers==0.38.0 gymnasium==1.3.0 torch torchvision hydra-core omegaconf tyro h5py tensorboard kagglehub
!pip install -q -e .
os.environ['DISPLAY'] = ''
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
""")

code(r"""
#@title 3. Demos: Drive cache → else Kaggle download (rgb h5, ~470 MB)
from google.colab import userdata
def _secret(k):
    try:
        return userdata.get(k)
    except Exception:
        return None
tok = _secret("KAGGLE_API_TOKEN")
if tok:
    os.environ["KAGGLE_API_TOKEN"] = tok
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
        f.write(tok)
    os.chmod(os.path.expanduser("~/.kaggle/access_token"), 0o600)
elif _secret("KAGGLE_USERNAME") and _secret("KAGGLE_KEY"):
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = _secret("KAGGLE_USERNAME"), _secret("KAGGLE_KEY")
else:
    print("WARNING: no Kaggle secret found; only the Drive cache can supply demos")

cached = glob.glob(f"{ROOT}/demos/*/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
if cached:
    print("copying cached demos from Drive (never train straight off the FUSE mount)")
    for h5 in cached:
        lvl = os.path.basename(os.path.dirname(h5)); os.makedirs(f"il/demos/{lvl}", exist_ok=True)
        for f in glob.glob(h5[:-2] + "*"):
            shutil.copy(f, f"il/demos/{lvl}/")
else:
    !python il/download_demos.py
    for h5 in glob.glob("il/demos/*/trajectory.*.h5"):
        lvl = os.path.basename(os.path.dirname(h5)); os.makedirs(f"{ROOT}/demos/{lvl}", exist_ok=True)
        for f in glob.glob(h5[:-2] + "*"):
            shutil.copy(f, f"{ROOT}/demos/{lvl}/")
    print("cached demos to Drive")
!ls -la il/demos/*/
""")

md(r"""
## 4. Verification cells (≈5 min, run once per fresh runtime)
Confirms rendering works, the scripted expert solves easy, the **action-clipping** fact
(demo actions up to |3.7| execute as ±1), and that the easy demos are (near-)identical trajectories.
""")

code(r"""
#@title 4a. Vulkan render + scripted expert sanity (expects 2/2)
import gymnasium as gym, torch, numpy as np, sys
sys.path.insert(0, '.')
import warehouse_sort
from examples.scripted_policy import scripted_episode
env = gym.make('WarehouseSort-v1', num_envs=1, obs_mode='rgb', control_mode='pd_ee_delta_pos', sim_backend='gpu',
               render_mode='rgb_array', difficulty='easy', num_parcels=2, fixed_poses=True, max_episode_steps=250)
obs, _ = env.reset(seed=0)
frame = env.render(); print("render ok:", tuple(frame.shape))
hist = scripted_episode(env, max_steps=250, seed=42)
print("scripted easy sorted:", hist[-1][-1]['success_count'].item(), "/ 2  in", len(hist), "steps")
env.close()
""")

code(r"""
#@title 4b. Action clipping check + demo statistics
import h5py
env = gym.make('WarehouseSort-v1', num_envs=1, obs_mode='state', control_mode='pd_ee_delta_pos', sim_backend='gpu',
               difficulty='easy', num_parcels=2, fixed_poses=True, max_episode_steps=250)
def disp(ax):
    env.reset(seed=0); p0 = env.unwrapped.agent.tcp_pose.p[0].clone()
    for _ in range(3):
        env.step(torch.tensor([[ax, 0., 0., 1.]]))
    return (env.unwrapped.agent.tcp_pose.p[0] - p0).cpu().numpy()
d_big, d_one = disp(3.66), disp(1.0)
print("TCP displacement a=3.66:", d_big.round(4), " a=1.0:", d_one.round(4), "-> identical:", np.allclose(d_big, d_one, atol=1e-4))
env.close()
with h5py.File(f'il/demos/{LEVEL}/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5', 'r') as f:
    keys = sorted(f.keys(), key=lambda k: int(k.split('_')[-1]))
    acts = [f[k]['actions'][()] for k in keys[:50]]
    over = np.mean([np.mean(np.abs(a[:, :3]) > 1.0) for a in acts])
    L = [a.shape[0] for a in acts]
    same = max(np.abs(acts[0] - a).max() for a in acts[1:] if a.shape == acts[0].shape) if len(acts) > 1 else 0
    print(f"{LEVEL}: {len(keys)} demos, len min/mean/max {min(L)}/{np.mean(L):.0f}/{max(L)}, "
          f"|xyz action|>1 in {over*100:.1f}% of steps, max|a_i - a_0| over first 50 demos = {same:.3f}")
""")

md(r"""
## 5. Smoke run (≈3 min): 300 iters on 20 demos → eval.py round trip → it/s
Fix `TOTAL_ITERS` from the measured speed **before** launching the real run (cosine LR depends on it):
30k iters must fit in ≈80 min.
""")

code(r"""
#@title 5. Smoke train + eval round trip
t0 = time.time()
!python il/train.py method=dp_rgb_{LEVEL} flags.exp_name=smoke flags.total_iters=300 flags.eval_freq=300 \
    flags.num_eval_episodes=8 flags.num_demos=20 flags.save_freq=100 flags.log_freq=50
print(f"smoke wall-clock {time.time()-t0:.0f}s (includes env + dataset setup)")
ck = "il/baselines/diffusion_policy/runs/smoke/checkpoints/final.pt"
!python eval.py difficulty={LEVEL} obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb checkpoint={ck} \
    eval_config=conf/eval/default.yaml record_video=false
""")

code(r"""
#@title 5b. Measure training speed (it/s) for TOTAL_ITERS planning
import re
log = open("il/baselines/diffusion_policy/runs/smoke/config.json").read()
# quick timing: 200 iters at full batch, no eval
t0 = time.time()
!python il/train.py method=dp_rgb_{LEVEL} flags.exp_name=speed flags.total_iters=200 flags.eval_freq=0 flags.save_freq=0 flags.log_freq=100 > /tmp/speed.log 2>&1
dt = time.time() - t0
print(open('/tmp/speed.log').read()[-600:])
print(f"≈{dt:.0f}s for 200 iters incl. ~40s setup -> projected 30k iters ≈ {(dt-40)/200*30000/60:.0f} min (rough)")
""")

md(r"""
## 6. Main training run (background, resumable)
Checkpoints go straight to Drive (`ROOT/ckpts/EXP/`): `latest.pt` every 2.5k iters,
`best_eval_sort_accuracy.pt` (+ `.submit.pt`) on every improvement. Logs → `ROOT/logs/`.
Re-running this cell after a VM recycle **resumes** automatically if `latest.pt` exists.
""")

code(r"""
#@title 6. Launch / resume training in the background
ckpt_dir = f"{ROOT}/ckpts/{EXP}"
os.makedirs(ckpt_dir, exist_ok=True)
resume = f"flags.resume={ckpt_dir}/latest.pt" if os.path.exists(f"{ckpt_dir}/latest.pt") else ""
log = f"{ROOT}/logs/{EXP}_{time.strftime('%m%d-%H%M')}.log"
cmd = (f"cd {REPO_DIR} && nohup python il/train.py method=dp_rgb_{LEVEL} flags.exp_name={EXP} "
       f"flags.total_iters={TOTAL_ITERS} flags.ckpt_dir={ckpt_dir} {resume} {EXTRA_FLAGS} > {log} 2>&1 &")
print(cmd)
subprocess.run(["bash", "-lc", cmd], check=True)
# background mirror of runs/ (tensorboard + results.json) to Drive every 5 min
subprocess.run(["bash", "-lc", f"cd {REPO_DIR} && nohup bash -c 'while true; do rsync -a --exclude videos il/baselines/diffusion_policy/runs/ {ROOT}/runs/; sleep 300; done' > /dev/null 2>&1 &"])
with open(f"{ROOT}/RUNLOG.md", "a") as f:
    f.write(f"- {time.strftime('%H:%M')} launched {EXP} ({'resume' if resume else 'fresh'}) log={os.path.basename(log)}\n")
print("launched; log:", log)
""")

code(r"""
#@title 6b. Monitor (re-run any time)
logs = sorted(glob.glob(f"{ROOT}/logs/{EXP}_*.log"), key=os.path.getmtime)
!tail -n 12 {logs[-1]}
!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
!free -h | head -2
rj = f"il/baselines/diffusion_policy/runs/{EXP}/results.json"
if os.path.exists(rj):
    r = json.load(open(rj)); print("best:", r["best"]); print("history:", [(h["iteration"], round(h.get("sort_accuracy", 0), 3)) for h in r["eval_history"]])
!ls -la {ROOT}/ckpts/{EXP}/ 2>/dev/null | tail -5
""")

md(r"""
## 7. Decision-grade evaluation (64 episodes on easy, 32 on medium/hard)
Eval-time knobs need **no retraining**: `act_horizon`, `num_inference_steps`, `scheduler` (ddpm/ddim),
`gripper_binarize`. Each eval appends a line to `ROOT/evals/results.jsonl`.
""")

code(r"""
#@title 7. Eval sweep on the best checkpoint
best = f"{ROOT}/ckpts/{EXP}/best_eval_sort_accuracy.pt"
if not os.path.exists(best):
    best = f"{ROOT}/ckpts/{EXP}/latest.pt"; print("no best yet -> using latest.pt")
evalcfg = "conf/eval/eval64.yaml" if LEVEL == "easy" else "conf/eval/eval32.yaml"
SWEEP = [dict(act_horizon=8, num_inference_steps=16), dict(act_horizon=4, num_inference_steps=16),
         dict(act_horizon=8, num_inference_steps=32), dict(act_horizon=4, num_inference_steps=32)]
for kw in SWEEP:
    ov = " ".join(f"+policy_kwargs.{k}={v}" for k, v in kw.items())
    print("\n=====", kw)
    !python eval.py difficulty={LEVEL} obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb checkpoint={best} \
        eval_config={evalcfg} record_video=false results_file={ROOT}/evals/results.jsonl {ov} 2>&1 | grep -E "SORT ACCURACY|mean_sorted|all_placed|mis_sort|eval_seconds|Error|error"
!tail -n 4 {ROOT}/evals/results.jsonl
""")

code(r"""
#@title 7b. Cross-level eval of one checkpoint (rgb obs has the same shape at every level)
for lvl in ["easy", "medium", "hard"]:
    print("\n=====", lvl)
    !python eval.py difficulty={lvl} obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb checkpoint={best} \
        eval_config=conf/eval/eval32.yaml record_video=false results_file={ROOT}/evals/results.jsonl 2>&1 | grep -E "SORT ACCURACY|mean_sorted|all_placed|mis_sort"
""")

code(r"""
#@title 8. Rollout video (1 env) + diagnostics
!python eval.py difficulty={LEVEL} obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb checkpoint={best} \
    eval_config=conf/eval/default.yaml record_video=true video_envs=1 2>&1 | tail -3
vids = sorted(glob.glob('outputs/**/videos/*.mp4', recursive=True), key=os.path.getmtime)
if vids:
    shutil.copy(vids[-1], f"{ROOT}/videos/{EXP}_{LEVEL}_{time.strftime('%m%d-%H%M')}.mp4")
    from IPython.display import Video, display
    display(Video(vids[-1], embed=True, width=640))
!python tools/rollout_logger.py difficulty={LEVEL} obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint={best} eval_config=conf/eval/eval32.yaml num_envs=8 +log_episodes=8 +log_out={ROOT}/evals/diag_{EXP}.json 2>&1 | tail -12
""")

md(r"""
## 9. Package + push (end of session)
Strips the best checkpoint to EMA weights + config (~30 MB), commits it under `checkpoints/`, updates
`submission.yaml` if you want this level to point at it, and pushes with the `GH_TOKEN` secret.
""")

code(r"""
#@title 9. Strip best checkpoint → checkpoints/, commit, push
gh = _secret("GH_TOKEN")
out = f"checkpoints/rgb_dp_{LEVEL}.pt"
!python tools/strip_ckpt.py {best} {out}
!git config user.email "mrpc2003@users.noreply.github.com" && git config user.name "mrpc2003"
!git add {out} submission.yaml docs/EXPERIMENTS.md 2>/dev/null; git commit -q -m "ckpt: {EXP} ({LEVEL}) from Colab" || echo "nothing to commit"
if gh:
    r = subprocess.run(["git", "push", f"https://mrpc2003:{gh}@github.com/mrpc2003/berlin-marso-hackathon.git", f"HEAD:{BRANCH}"], capture_output=True, text=True)
    print("push:", "ok" if r.returncode == 0 else r.stderr[-500:])
else:
    print("no GH_TOKEN secret -> commit is local only (checkpoint is also on Drive)")
with open(f"{ROOT}/RUNLOG.md", "a") as f:
    f.write(f"- {time.strftime('%H:%M')} session end: stripped {best} -> {out}\n")
drive.flush_and_unmount(); print("Drive flushed")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open("MARSO_COLAB.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote MARSO_COLAB.ipynb with {len(cells)} cells")
