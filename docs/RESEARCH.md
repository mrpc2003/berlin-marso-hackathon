# Research notes — what participants found, what we measured

Sources: Kaggle writeups (CC BY 4.0) and public forks of the starter repo; our own measurements on the
competition data and the ManiSkill 3.0.1 source. Competition:
https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge

## Leaderboard reality
- 4 submissions. 1st place ("Glasses On", Sahil Rajpurkar) used **ACT on rgb**: easy 0.50 / medium 0.22 /
  hard 0.083 → weighted ≈ 0.21. Their state DP and SAC reached 0.5 on easy only.
- Team Blades (state DP): easy ~50–62%, medium ~9–14%, no hard. Marso-Hacklab-Waitlist (iambodha fork,
  state DP + Gaussian state noise): easy 53%, medium 20%, no hard. FOYSAL (rgb DP template): 0.
- **No participant produced a verified nonzero score with a pure-rgb Diffusion Policy**, and nobody
  submitted a hard checkpoint on the state track.
- Scoring is per parcel: sorting only the FIRST parcel reliably on every level already gives
  0.2·0.5 + 0.3·0.25 + 0.5·0.167 = 0.258 > 0.21.

## Structural problems everyone hit (fixed in this fork)
1. **Deployment ≠ training-time evaluation.** `_DPRgbPolicy.act` re-sampled a fresh diffusion plan every
   step and executed only its first action (and faked the observation history); the in-training evaluator
   executes `act_horizon` actions per plan through `FrameStack`. The gripper command flickered between
   independent samples and never latched. Fix: `warehouse_sort/il_policy.py` (`ChunkedPolicy`).
   Reported effect on state DP easy: 0% → 50–62% (Team Blades).
2. **Episode budget.** Demos take ~115 / 234–238 / 348–779 steps (easy/medium/hard, measured from the
   trajectory json) but `max_episode_steps` was 200 everywhere, so medium/hard could not finish even when
   imitated perfectly. Placement is sticky in `env.evaluate()` (`|=`), so a longer budget never hurts.
   Fix: per-difficulty `difficulty.max_episode_steps` (250/500/800) + interpolation in `conf/config.yaml`
   (Hydra loads `difficulty` before `_self_`, so a literal there would silently win).
3. **Hyperparameter drift between trainer and loader** (pred_horizon 32 vs 16, 100 vs 16 denoising steps).
   Fix: every checkpoint carries a `config` dict; loaders rebuild from it.
4. **Noisy model selection**: 4–16 episode evals. Fix: 32 in training, 64 for decisions on easy.

## What we measured ourselves
- **Demo actions exceed the action space**: `actions[:, :3]` reach |3.66| (the scripted `_move` divides
  by 0.1 without clipping). ManiSkill's `gym_utils.clip_and_scale_action` clips to [-1, 1] before
  scaling, so the executed action was clip(a). Diffusion Policy trains with `clip_sample=True`, i.e. it
  can never reproduce the stored target. Fix: `clip_actions` in the loaders (default on). ACT uses an
  L1 loss without clipping — consistent with ACT being the only rgb method that worked.
- **Easy demos are ~200 copies of one trajectory**: fixed scene + deterministic expert + the env
  overrides the robot init with exact `START_QPOS` (so `robot_init_qpos_noise` never applies). No
  recovery data → the second parcel starts off-distribution. Mitigation: DART-style demos
  (`il/gen_demos.py --action-noise 0.05`), random-shift augmentation, proprio noise.
- **Memory**: `capture_video=true` makes `RecordEpisode` buffer whole episodes of tiled 512-px frames in
  system RAM (≈5 GB on hard) — the actual OOM risk on Colab, not the eval envs. The stock loader also
  materialises the full h5 in RAM (7 GB for hard). Fix: `capture_video=false`, 1-env eval video,
  streaming per-trajectory loader with the dataset resident on the GPU as uint8 (easy 1.13 / medium
  2.33 / hard 3.82 GB).
- The "SpatialSoftmax is colour-blind" diagnosis (patcadragos fork) is overstated: tag colour is a
  deterministic function of the grid slot (`tags = [i % 2]`), demos pick slots in index order, and the
  bins are ~31×37 px — keypoint channels on ResNet features can localise a red blob. Kept only as a
  contingency.
- The parcel grid depends on the parcel count (2 parcels sit at y=0; with 6 the first row is at
  y=-0.112), so a hard-only rgb policy would reach for empty slots on easy → per-level checkpoints.

## Participant patches we ported (with thanks)
- chunked deployment: tylertan-tech, saribx, harshangs (Team Blades), iambodha
- DrQ random-shift aug: tylertan-tech, patcadragos
- resume / config-in-checkpoint / eval_inference_steps / state noise: iambodha
- per-difficulty episode length + ACT baseline vendored for WarehouseSort: sahilrajpurkar03
  (`hackathon-dev-act-merged`), used as the hedge line in `il/baselines/act/`

## Colab T4 facts
- `pip install mani-skill==3.0.1 diffusers==0.38.0 gymnasium torch torchvision hydra-core kagglehub`,
  `DISPLAY=''`, `PYOPENGL_PLATFORM=egl` — Vulkan rendering works on the free T4.
- Kaggle: new-format token in `~/.kaggle/access_token` (or `KAGGLE_API_TOKEN`); downloads still work
  after the competition closed.
- Two participant notebooks committed Kaggle keys in plain text. Use Colab Secrets only.
