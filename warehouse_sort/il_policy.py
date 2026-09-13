"""Imitation-learning policy entrypoints for eval.py / the judge.

Each loader satisfies the policy contract:
    load_fn(checkpoint, sample_obs, action_space, device) -> policy
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    python eval.py difficulty=easy obs_mode=rgb \\
        policy=warehouse_sort.il_policy:load_dp_rgb \\
        checkpoint=<path> eval_config=conf/eval/eval32.yaml

  load_dp      -- state Diffusion Policy (one checkpoint PER level; state dim depends on parcels)
  load_dp_rgb  -- RGB Diffusion Policy (scene image + proprioception; one ckpt can serve all levels)

Deployment matches the training-time evaluator (diffusion_policy/evaluate.py + FrameStack):
  * a real rolling history of the last `obs_horizon` observations conditions the model, and
  * each diffusion call yields `act_horizon` actions that are executed before re-planning
    (receding-horizon chunking). The original template re-sampled a fresh plan every step and
    executed only its first action, which made the gripper command flicker between independent
    samples and never latch a grasp.

Architecture / horizon hyperparameters are read from the checkpoint's `config` dict (written by
the trainers), so the judge can call the loader with no extra kwargs. Explicit kwargs override.
"""

from collections import deque
from types import SimpleNamespace

import torch

from warehouse_sort.constants import GRIPPER_INDEX, QPOS_SLICE, START_QPOS

# --------------------------------------------------------------------------- #
# defaults (must match il/conf/method/dp*.yaml when no `config` is stored in the checkpoint)
# --------------------------------------------------------------------------- #
DEFAULT_DP_CONFIG = dict(
    obs_horizon=2,
    act_horizon=8,
    pred_horizon=16,
    diffusion_step_embed_dim=64,
    unet_dims=[64, 128, 256],
    n_groups=8,
    num_diffusion_iters=100,
    num_inference_steps=16,   # eval-only knob (DDPM steps); training always uses num_diffusion_iters
    scheduler="ddpm",         # "ddpm" | "ddim" (eval-only)
    gripper_binarize=False,   # snap the gripper command to +/-1 at deployment (eval-only)
    seed=0,                   # seed for the diffusion noise generator (reproducible evals)
    amp_eval=False,           # fp16 autocast for the visual encoder at eval (rgb only)
    # rgb-only
    visual_encoder="resnet18",
    num_kp=32,
)


def _add_baseline_path(rel):
    import os
    import sys

    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_policy_config(ckpt, overrides=None, defaults=DEFAULT_DP_CONFIG):
    """defaults < checkpoint['config'] < explicit (non-None) overrides."""
    cfg = dict(defaults)
    stored = ckpt.get("config") if isinstance(ckpt, dict) else None
    if isinstance(stored, dict):
        cfg.update({k: v for k, v in stored.items() if v is not None})
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    if isinstance(cfg.get("unet_dims"), (tuple, list)):
        cfg["unet_dims"] = [int(x) for x in cfg["unet_dims"]]
    return cfg


def _load_state_dict(ckpt):
    """Accept full trainer checkpoints ({'agent','ema_agent',...}) and stripped submission
    checkpoints ({'model': ..., 'config': ...}). EMA weights are preferred."""
    for key in ("ema_agent", "model", "agent"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    raise KeyError(f"no weights found in checkpoint (keys: {list(ckpt.keys())})")


def make_scheduler(kind, num_train_timesteps):
    """Same noise schedule as training (squaredcos_cap_v2, epsilon, clip_sample)."""
    if kind == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        return DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2",
                             clip_sample=True, prediction_type="epsilon")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def diffusion_sample(net, scheduler, obs_cond, pred_horizon, act_dim, generator=None):
    """Reverse-diffuse an action chunk (B, pred_horizon, act_dim) conditioned on obs_cond."""
    B = obs_cond.shape[0]
    device = obs_cond.device
    naction = torch.randn((B, pred_horizon, act_dim), device=device, generator=generator)
    for k in scheduler.timesteps:
        noise_pred = net(sample=naction, timestep=k, global_cond=obs_cond)
        # generator also drives the DDPM variance noise inside step(); without it evals are not reproducible
        naction = scheduler.step(model_output=noise_pred, timestep=k, sample=naction, generator=generator).prev_sample
    return naction


class ChunkedPolicy:
    """Receding-horizon executor with rolling observation history and episode-reset detection.

    Subclasses implement `_prep(obs) -> dict[str, Tensor]` (tensors on self.device, batch first)
    and `_plan(obs_seq) -> Tensor (B, act_horizon, act_dim)` where obs_seq stacks the history along
    dim 1. `reset()` is called by warehouse_sort.utils before every episode batch; because the judge
    might run a harness that never calls it, a reset is ALSO detected from proprioception: a fresh
    episode starts at the exact home configuration (START_QPOS) or teleports the joints.
    """

    def __init__(self, obs_horizon, act_horizon, device, gripper_binarize=False,
                 home_tol=0.03, jump_tol=0.3):
        self.obs_horizon = int(obs_horizon)
        self.act_horizon = int(act_horizon)
        self.device = torch.device(device)
        self.gripper_binarize = bool(gripper_binarize)
        self.home_tol = float(home_tol)
        self.jump_tol = float(jump_tol)
        self._home = torch.tensor(START_QPOS[:7], dtype=torch.float32, device=self.device)
        self.reset()

    # -- episode bookkeeping ------------------------------------------------------------- #
    def reset(self):
        self._hist = None      # deque of prepared obs dicts (len <= obs_horizon)
        self._queue = []       # not-yet-executed actions from the last plan
        self._batch = None     # batch size of the current rollout
        self._prev_qpos = None
        self.n_plans = 0

    def _at_home(self, qpos7):
        return (qpos7 - self._home).abs().max(dim=1).values < self.home_tol

    def _detect_reset(self, qpos7):
        if self._prev_qpos is None or self._prev_qpos.shape != qpos7.shape:
            return True
        returned_home = self._at_home(qpos7) & ~self._at_home(self._prev_qpos)
        jumped = (qpos7 - self._prev_qpos).abs().max(dim=1).values > self.jump_tol
        return bool((returned_home | jumped).any())

    def _stack(self):
        keys = self._hist[0].keys()
        return {k: torch.stack([h[k] for h in self._hist], dim=1) for k in keys}

    # -- contract ------------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, deterministic=True):
        cur = self._prep(obs)
        qpos7 = cur["state"][:, QPOS_SLICE][:, :7].float()
        B = qpos7.shape[0]
        if self._hist is None or self._batch != B or self._detect_reset(qpos7):
            self._hist = deque([cur] * self.obs_horizon, maxlen=self.obs_horizon)
            self._queue = []
            self._batch = B
        else:
            self._hist.append(cur)
        self._prev_qpos = qpos7

        if not self._queue:
            plan = self._plan(self._stack()).clamp(-1.0, 1.0)      # (B, act_horizon, act_dim)
            self._queue = list(plan.unbind(dim=1))
            self.n_plans += 1
        a = self._queue.pop(0)
        if self.gripper_binarize:
            a = a.clone()
            a[:, GRIPPER_INDEX] = torch.where(a[:, GRIPPER_INDEX] >= 0, 1.0, -1.0)
        return a


# --------------------------------------------------------------------------- #
# State Diffusion Policy (privileged low-dim obs; one checkpoint per level)
# --------------------------------------------------------------------------- #
class _DPPolicy(ChunkedPolicy):
    def __init__(self, net, scheduler, cfg, act_dim, device):
        super().__init__(cfg["obs_horizon"], cfg["act_horizon"], device, cfg.get("gripper_binarize", False))
        self.net = net.to(device).eval()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(int(cfg["num_inference_steps"]))
        self.pred_horizon = int(cfg["pred_horizon"])
        self.act_dim = int(act_dim)
        self.generator = torch.Generator(device=self.device).manual_seed(int(cfg.get("seed", 0)))

    def _prep(self, obs):
        state = obs["state"] if isinstance(obs, dict) else obs
        return {"state": state.float().to(self.device)}

    def _plan(self, obs_seq):
        obs_cond = obs_seq["state"].flatten(start_dim=1)          # (B, obs_horizon * obs_dim)
        naction = diffusion_sample(self.net, self.scheduler, obs_cond, self.pred_horizon, self.act_dim,
                                   generator=self.generator)
        start = self.obs_horizon - 1
        return naction[:, start:start + self.act_horizon]


def load_dp(checkpoint, sample_obs, action_space, device, **overrides):
    """Load a state Diffusion Policy checkpoint (EMA weights). Hyperparameters come from the
    checkpoint's `config`; kwargs override (e.g. act_horizon=4, num_inference_steps=32)."""
    _add_baseline_path("diffusion_policy")
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = resolve_policy_config(ckpt, overrides)
    state = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
    obs_dim = state.shape[1]
    act_dim = action_space.shape[0]
    net = ConditionalUnet1D(
        input_dim=act_dim, global_cond_dim=cfg["obs_horizon"] * obs_dim,
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=list(cfg["unet_dims"]), n_groups=cfg["n_groups"],
    )
    sd = _load_state_dict(ckpt)
    net_sd = {k.replace("noise_pred_net.", "", 1): v for k, v in sd.items() if k.startswith("noise_pred_net.")}
    net.load_state_dict(net_sd)
    scheduler = make_scheduler(cfg.get("scheduler", "ddpm"), cfg["num_diffusion_iters"])
    return _DPPolicy(net, scheduler, cfg, act_dim, device)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy (scene image + proprioception; NO privileged state)
# --------------------------------------------------------------------------- #
class _DPRgbPolicy(ChunkedPolicy):
    def __init__(self, agent, cfg, device):
        super().__init__(cfg["obs_horizon"], cfg["act_horizon"], device, cfg.get("gripper_binarize", False))
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler = make_scheduler(cfg.get("scheduler", "ddpm"), cfg["num_diffusion_iters"])
        self.agent.noise_scheduler.set_timesteps(int(cfg["num_inference_steps"]))
        self.agent.amp = bool(cfg.get("amp_eval", False))
        self.generator = torch.Generator(device=self.device).manual_seed(int(cfg.get("seed", 0)))
        self.cfg = cfg

    def _prep(self, obs):
        return {"state": obs["state"].float().to(self.device), "rgb": obs["rgb"].to(self.device)}

    def _plan(self, obs_seq):
        # obs_seq["rgb"]: (B, obs_horizon, H, W, 3) uint8 -- Agent.get_action permutes to channels-first
        return self.agent.get_action(obs_seq, generator=self.generator)   # (B, act_horizon, act_dim)


def load_dp_rgb(checkpoint, sample_obs, action_space, device, **overrides):
    """Load an RGB Diffusion Policy checkpoint (vendored train_rgbd.Agent; EMA weights)."""
    import numpy as np
    import gymnasium.spaces as spaces

    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = resolve_policy_config(ckpt, overrides)
    h, w, c = sample_obs["rgb"].shape[1:]
    state_dim = sample_obs["state"].shape[1]
    oh = int(cfg["obs_horizon"])
    stub = SimpleNamespace(
        single_observation_space=spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (oh, state_dim), np.float32),
            "rgb": spaces.Box(0, 255, (oh, h, w, c), np.uint8),
        }),
        single_action_space=spaces.Box(-1.0, 1.0, (action_space.shape[0],), np.float32),
    )
    args = SimpleNamespace(
        obs_horizon=oh, act_horizon=int(cfg["act_horizon"]), pred_horizon=int(cfg["pred_horizon"]),
        diffusion_step_embed_dim=int(cfg["diffusion_step_embed_dim"]), unet_dims=list(cfg["unet_dims"]),
        n_groups=int(cfg["n_groups"]), visual_encoder=cfg["visual_encoder"], num_kp=int(cfg["num_kp"]),
        image_aug_pad=0, proprio_noise_std=0.0, amp=False,
    )
    agent = Agent(stub, args)
    agent.load_state_dict(_load_state_dict(ckpt))
    return _DPRgbPolicy(agent, cfg, device)


# Convenience entrypoints for submission.yaml (the judge passes no kwargs). Add more as needed.
def load_dp_rgb_ah4(checkpoint, sample_obs, action_space, device, **kw):
    kw.setdefault("act_horizon", 4)
    return load_dp_rgb(checkpoint, sample_obs, action_space, device, **kw)
