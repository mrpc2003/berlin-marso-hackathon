"""RGB Diffusion Policy trainer for WarehouseSort (vendored ManiSkill baseline, reworked for Colab T4).

Changes vs. the ManiSkill template (see docs/RESEARCH.md for why):
  * streaming per-trajectory demo loader (StreamingRGBDemoDataset): whole dataset resident on the
    GPU as uint8, system-RAM peak ~= one trajectory; multiple --demo-path files = mixed-level training
  * demo actions clipped to [-1, 1] (the controller executes clip(a); the h5 stores unclipped deltas)
  * training-only DrQ random-shift augmentation (--image-aug-pad) and proprio noise
  * fp16 autocast on the visual encoder (--amp) with GradScaler; --no-torch-deterministic for speed
  * every checkpoint stores a policy `config` (horizons, encoder, ...) so warehouse_sort.il_policy
    can rebuild the model without CLI flags; best checkpoints also get a stripped `*.submit.pt`
  * --resume from latest.pt (agent, EMA, optimizer, LR schedule, scaler, RNG, best metrics, history)
  * --ckpt-dir (e.g. a Google Drive path) and --save-freq for latest.pt
  * the training-time evaluator uses --eval-inference-steps (16) like deployment, skips the useless
    iteration-0 eval, and --eval-freq 0 disables sim eval entirely (no ManiSkill import needed)
  * capture_video defaults to False (RecordEpisode buffers full episodes of frames in system RAM)

  python train_rgbd.py --demo-path ../../demos/easy/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5 \
      --env-id WarehouseSort-v1 --control-mode pd_ee_delta_pos --sim-backend gpu --max-episode-steps 250
Normally launched through il/train.py (Hydra) -- see il/conf/method/dp_rgb_*.yaml.
"""

ALGO_NAME = "BC_Diffusion_rgb_UNet"

import json
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusion_policy.augment import RandomShiftsAug
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.streaming_dataset import StreamingRGBDemoDataset, demo_env_info
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment (run dir runs/<exp_name>)"""
    exp_name_timestamp: bool = False
    """append _MMDD-HHMM to exp_name so a re-run never overwrites a better run's best checkpoint"""
    seed: int = 1
    torch_deterministic: bool = True
    """cudnn.deterministic; pass --no-torch-deterministic for cudnn.benchmark speed on T4"""
    cuda: bool = True
    capture_video: bool = False
    """record eval rollouts during training. OFF by default: RecordEpisode buffers whole episodes of
    tiled 512px frames in system RAM (several GB on hard) and OOM-kills a 12.7 GB Colab runtime."""

    env_id: str = "WarehouseSort-v1"
    demo_path: List[str] = field(default_factory=list)
    """one or more ManiSkill rgb demo .h5 files (each with its .json next to it). The FIRST file's
    recorded env kwargs define the training-time eval scene (put the level you want to track first)."""
    num_demos: Optional[int] = None
    """max trajectories to load PER file"""
    total_iters: int = 30_000
    batch_size: int = 128

    # Diffusion Policy
    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8
    num_diffusion_iters: int = 100
    """DDPM training timesteps"""
    eval_inference_steps: int = 16
    """denoising steps for the training-time evaluator AND the default stored for deployment"""

    # observation / encoder
    obs_mode: str = "rgb"
    obs_camera: str = "scene"
    visual_encoder: str = "resnet18"
    """"resnet18" (ResNet18 trunk + SpatialSoftmax keypoints) or "plain_conv" (flattened conv map)"""
    num_kp: int = 32

    # data / regularisation
    clip_actions: bool = True
    """clip demo actions to [-1, 1] = what the controller actually executed"""
    image_aug_pad: int = 4
    """DrQ random-shift augmentation pad in px (0 = off); training only"""
    proprio_noise_std: float = 0.0
    """Gaussian noise on the 26-d proprioception during training (0 = off)"""
    amp: bool = True
    """fp16 autocast for the visual encoder (UNet stays fp32); needs CUDA"""

    # environment / experiment
    num_parcels: int = 2
    max_episode_steps: Optional[int] = None
    """episode budget for the training-time eval env (must cover the demo length: ~115/236/388)"""
    log_freq: int = 500
    eval_freq: int = 5000
    """evaluate every N iters; 0 = never (no simulator needed -> CPU smoke tests)"""
    skip_initial_eval: bool = True
    save_freq: int = 2500
    """write latest.pt every N iters (resume point)"""
    num_eval_episodes: int = 32
    num_eval_envs: int = 8
    sim_backend: str = "gpu"
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_delta_pos"
    resume: Optional[str] = None
    """path to a latest.pt to continue from (same hyperparameters!)"""
    ckpt_dir: Optional[str] = None
    """where checkpoints go (default runs/<exp_name>/checkpoints); point at Drive on Colab"""
    demo_type: Optional[str] = None


# --------------------------------------------------------------------------- #
class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        obs_space = env.single_observation_space
        assert len(obs_space["state"].shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = obs_space["state"].shape[1]
        assert "rgb" in obs_space.keys(), "this trainer is rgb-only"
        self.include_rgb = True
        self.include_depth = False
        total_visual_channels = obs_space["rgb"].shape[-1]

        visual_feature_dim = 256
        enc = getattr(args, "visual_encoder", "resnet18")
        if enc == "resnet18":
            from diffusion_policy.lerobot_encoder import ResNet18SpatialSoftmax
            self.visual_encoder = ResNet18SpatialSoftmax(
                in_channels=total_visual_channels, out_dim=visual_feature_dim, num_kp=getattr(args, "num_kp", 32))
        elif enc == "plain_conv":
            self.visual_encoder = PlainConv(in_channels=total_visual_channels, out_dim=visual_feature_dim,
                                            pool_feature_map=False)
        else:
            raise ValueError(f"unknown visual_encoder {enc!r} (resnet18 | plain_conv)")
        pad = int(getattr(args, "image_aug_pad", 0) or 0)
        if pad > 0:
            self.aug = RandomShiftsAug(pad)      # parameter-free, training only
        self.proprio_noise_std = float(getattr(args, "proprio_noise_std", 0.0) or 0.0)
        self.amp = bool(getattr(args, "amp", False))

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=list(args.unet_dims),
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = int(getattr(args, "num_diffusion_iters", 100))
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,                   # clip output to [-1,1] to improve stability
            prediction_type="epsilon",          # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        rgb = obs_seq["rgb"]
        if rgb.shape[-1] == 3 and rgb.shape[2] != 3:          # (B, T, H, W, C) from the env -> channels first
            rgb = rgb.permute(0, 1, 4, 2, 3)
        rgb = rgb.float() / 255.0                             # (B, obs_horizon, C, H, W)
        B = rgb.shape[0]
        img = rgb.flatten(end_dim=1)                          # (B*obs_horizon, C, H, W)
        if not eval_mode and hasattr(self, "aug"):
            img = self.aug(img)
        with torch.autocast(device_type=img.device.type, dtype=torch.float16,
                            enabled=self.amp and img.device.type == "cuda"):
            feat = self.visual_encoder(img)
        feat = feat.float().reshape(B, self.obs_horizon, -1)  # (B, obs_horizon, D)
        state = obs_seq["state"].float()
        if not eval_mode and self.proprio_noise_std > 0:
            state = state + torch.randn_like(state) * self.proprio_noise_std
        return torch.cat((feat, state), dim=-1).flatten(start_dim=1)   # (B, obs_horizon*(D+state))

    def compute_loss(self, obs_seq, action_seq):
        B = action_seq.shape[0]
        device = action_seq.device
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def get_action(self, obs_seq, generator=None):
        """obs_seq: {"rgb": (B, obs_horizon, H, W, C) uint8 or (B, obs_horizon, C, H, W), "state": (B, obs_horizon, D)}
        -> (B, act_horizon, act_dim). Denoising uses whatever self.noise_scheduler.set_timesteps() set."""
        obs_cond = self.encode_obs(obs_seq, eval_mode=True)
        B = obs_cond.shape[0]
        naction = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_cond.device, generator=generator)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction,
                                                generator=generator).prev_sample   # DDPM variance noise too
        start = self.obs_horizon - 1
        return naction[:, start:start + self.act_horizon]


# --------------------------------------------------------------------------- #
def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _cpu_rng_state(state):
    """torch.set_rng_state expects a contiguous CPU uint8 tensor even after CUDA map_location loads."""
    return state.detach().cpu().contiguous().to(dtype=torch.uint8)


def policy_config(args, state_dim, act_dim, image_hw, demo_paths):
    """What warehouse_sort.il_policy.load_dp_rgb needs to rebuild + run this model (stored in every ckpt)."""
    return dict(
        obs_horizon=args.obs_horizon, act_horizon=args.act_horizon, pred_horizon=args.pred_horizon,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim, unet_dims=list(args.unet_dims),
        n_groups=args.n_groups, num_diffusion_iters=args.num_diffusion_iters,
        num_inference_steps=args.eval_inference_steps, scheduler="ddpm",
        visual_encoder=args.visual_encoder, num_kp=args.num_kp, obs_mode=args.obs_mode,
        state_dim=int(state_dim), act_dim=int(act_dim), image_hw=list(image_hw),
        gripper_binarize=False, amp_eval=False, seed=0,
        # provenance
        algo=ALGO_NAME, exp_name=args.exp_name, clip_actions=args.clip_actions,
        image_aug_pad=args.image_aug_pad, proprio_noise_std=args.proprio_noise_std,
        max_episode_steps=args.max_episode_steps, demo_paths=[os.path.abspath(p) for p in demo_paths],
        git=_git_hash(), lr=args.lr, batch_size=args.batch_size, total_iters=args.total_iters,
    )


class _MockEnv:
    """Observation/action spaces derived from the dataset (used when --eval-freq 0: no simulator)."""

    def __init__(self, dataset, obs_horizon):
        import gymnasium.spaces as spaces
        tr = dataset.trajectories[0]
        _, c, h, w = tr["rgb"].shape
        d = tr["state"].shape[1]
        a = tr["actions"].shape[1]
        self.single_observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, d), np.float32),
            "rgb": spaces.Box(0, 255, (obs_horizon, h, w, c), np.uint8),
        })
        self.single_action_space = spaces.Box(-1.0, 1.0, (a,), np.float32)

    def close(self):
        pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    demo_paths = list(args.demo_path)
    assert demo_paths, "pass at least one --demo-path <file.h5>"
    if args.exp_name is None:
        args.exp_name = f"{args.env_id}__{ALGO_NAME}__{args.seed}__{int(time.time())}"
    if args.exp_name_timestamp:
        args.exp_name += time.strftime("_%m%d-%H%M")
    run_name = args.exp_name
    run_dir = os.path.join("runs", run_name)
    ckpt_dir = args.ckpt_dir or os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    env_info = demo_env_info(demo_paths[0])
    demo_kwargs = env_info["env_kwargs"]
    control_mode = demo_kwargs.get("control_mode") or env_info.get("episodes", [{}])[0].get("control_mode")
    assert control_mode == args.control_mode, f"control mode mismatch: dataset {control_mode} vs args {args.control_mode}"
    scene_kwargs = {k: demo_kwargs[k] for k in ("num_parcels", "fixed_poses", "randomization", "obs_camera") if k in demo_kwargs}
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    assert args.max_episode_steps is not None or args.eval_freq <= 0, "max_episode_steps is required when evaluating"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if not args.torch_deterministic:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    eval_enabled = args.eval_freq is not None and args.eval_freq > 0

    envs = None
    if eval_enabled:
        from diffusion_policy.make_env import make_eval_envs
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode=args.obs_mode,
                          obs_camera=args.obs_camera, render_mode="rgb_array",
                          human_render_camera_configs=dict(shader_pack="default"), num_parcels=args.num_parcels)
        env_kwargs.update(scene_kwargs)                      # demo-recorded kwargs win (exact distribution match)
        env_kwargs["max_episode_steps"] = args.max_episode_steps
        envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs,
                              dict(obs_horizon=args.obs_horizon),
                              video_dir=os.path.join(run_dir, "videos") if args.capture_video else None,
                              wrappers=[FlattenRGBDObservationWrapper])

    writer = SummaryWriter(run_dir)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])))

    t0 = time.time()
    dataset = StreamingRGBDemoDataset(demo_paths, args.obs_horizon, args.pred_horizon, device,
                                      num_demos=args.num_demos, clip_actions=args.clip_actions)
    print(f"[train_rgbd] dataset loaded in {time.time() - t0:.0f}s", flush=True)
    agent_env = envs if eval_enabled else _MockEnv(dataset, args.obs_horizon)

    agent = Agent(agent_env, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.total_iters)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(agent_env, args).to(device)
    ema_agent.noise_scheduler.set_timesteps(args.eval_inference_steps)   # deployment-like eval (16 steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    tr0 = dataset.trajectories[0]
    cfg = policy_config(args, tr0["state"].shape[1], tr0["actions"].shape[1], tr0["rgb"].shape[2:], demo_paths)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policy": cfg, "args": vars(args)}, f, indent=2, default=str)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[train_rgbd] run={run_name} device={device} params={n_params / 1e6:.2f}M ckpt_dir={ckpt_dir}", flush=True)

    best_eval_metrics = defaultdict(float)
    eval_history = []
    start_iteration = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        ema_agent.load_state_dict(ck["ema_agent"])
        optimizer.load_state_dict(ck["optimizer"])
        lr_scheduler.load_state_dict(ck["lr_scheduler"])
        ema.load_state_dict(ck["ema_state"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        best_eval_metrics.update(ck.get("best_eval_metrics", {}))
        eval_history = list(ck.get("eval_history", []))
        rng = ck.get("rng")
        if rng:
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(_cpu_rng_state(rng["torch"]))
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([_cpu_rng_state(s) for s in rng["cuda"]])
        start_iteration = int(ck.get("iteration", -1)) + 1
        print(f"[train_rgbd] resumed {args.resume} at iteration {start_iteration}/{args.total_iters}", flush=True)
        if start_iteration >= args.total_iters:
            print("[train_rgbd] nothing left to train (raise --total-iters to continue)")
            writer.close()
            raise SystemExit(0)

    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters, start_iter=start_iteration)
    train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_dataload_workers,
                                  worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=args.seed))

    def save_ckpt(tag, iteration, submit=False):
        ema.copy_to(ema_agent.parameters())
        payload = {
            "agent": agent.state_dict(), "ema_agent": ema_agent.state_dict(),
            "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(),
            "ema_state": ema.state_dict(), "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "iteration": int(iteration), "config": cfg, "args": vars(args),
            "best_eval_metrics": dict(best_eval_metrics), "eval_history": eval_history,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)      # atomic: a disconnect mid-write never corrupts latest.pt
        if submit:
            torch.save({"model": ema_agent.state_dict(), "config": cfg, "iteration": int(iteration)},
                       os.path.join(ckpt_dir, f"{tag}.submit.pt"))
        return path

    def evaluate_and_save_best(iteration):
        if not eval_enabled:
            return
        from diffusion_policy.evaluate import evaluate
        tick = time.time()
        ema.copy_to(ema_agent.parameters())
        m = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
        n_eps = len(m["success_at_end"]) if "success_at_end" in m else args.num_eval_episodes
        m = {k: float(np.mean(v)) for k, v in m.items()}
        for k, v in m.items():
            writer.add_scalar(f"eval/{k}", v, iteration)
        eval_history.append({"iteration": int(iteration), **m, "n_episodes": int(n_eps)})
        print(f"[eval @ {iteration}] {n_eps} eps in {time.time() - tick:.0f}s: "
              + " ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
        for k in ("sort_accuracy", "success_once", "success_at_end"):
            if k in m and m[k] > best_eval_metrics[k]:
                best_eval_metrics[k] = m[k]
                save_ckpt(f"best_eval_{k}", iteration, submit=(k == "sort_accuracy"))
                print(f"  new best {k}={m[k]:.4f} -> saved", flush=True)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump({"exp_name": run_name, "best": dict(best_eval_metrics), "eval_history": eval_history}, f, indent=1)

    agent.train()
    pbar = tqdm(total=args.total_iters, initial=start_iteration)
    timings = defaultdict(float)
    last_tick = time.time()
    total_loss = torch.zeros(())
    iteration = start_iteration
    for iteration, data_batch in enumerate(train_dataloader, start=start_iteration):
        timings["data"] += time.time() - last_tick
        tick = time.time()
        total_loss = agent.compute_loss(obs_seq=data_batch["observations"], action_seq=data_batch["actions"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        ema.step(agent.parameters())
        timings["step"] += time.time() - tick

        if eval_enabled and iteration % args.eval_freq == 0 and (iteration > 0 or not args.skip_initial_eval):
            evaluate_and_save_best(iteration)
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)
        if args.save_freq and iteration % args.save_freq == 0 and iteration > start_iteration:
            save_ckpt("latest", iteration)
        pbar.update(1)
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        last_tick = time.time()

    pbar.close()   # explicit: tqdm.__del__ during interpreter teardown can segfault
    final_iter = args.total_iters
    evaluate_and_save_best(final_iter)
    save_ckpt("latest", final_iter)
    save_ckpt("final", final_iter, submit=True)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"exp_name": run_name, "best": dict(best_eval_metrics), "eval_history": eval_history,
                   "timings": dict(timings), "params_M": n_params / 1e6}, f, indent=1)
    print(f"[train_rgbd] done. best={dict(best_eval_metrics)} ckpts in {ckpt_dir}", flush=True)
    if envs is not None:
        envs.close()
    writer.close()
