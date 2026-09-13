"""Memory-lean demo loader for the RGB Diffusion Policy.

The vendored `load_demo_dataset` materialises EVERY key of the h5 (rgb frames, env states, camera
params, ...) for all trajectories in system RAM before anything reaches the GPU. On Colab
(12.7 GB RAM) that is ~7 GB for the hard set and OOM for mixed-level training. This loader reads
one trajectory at a time, keeps only what the policy consumes and moves it to `device`
immediately (rgb as uint8, so the whole hard set is ~3.8 GB of VRAM).

It also applies two data fixes:
  * `clip_actions`: the scripted demos store unclipped end-effector deltas (|a| up to ~3.7) while
    the controller executes clip(a, -1, 1). Training on the executed action keeps the diffusion
    targets inside the clip_sample range.
  * explicit proprioception key order (qpos, qvel, tcp_pose, is_grasped) matching
    FlattenRGBDObservationWrapper, independent of h5py's alphabetical key order.

Multiple demo files may be given (mixed-level training); trajectories are simply concatenated.
"""

import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

RGB_KEY = "obs/sensor_data/scene_camera/rgb"
STATE_KEYS = ("obs/agent/qpos", "obs/agent/qvel", "obs/extra/tcp_pose", "obs/extra/is_grasped")


def _traj_keys(f):
    return sorted((k for k in f.keys() if k.startswith("traj_")), key=lambda x: int(x.split("_")[-1]))


def read_trajectory(traj, device, clip_actions=True):
    """h5 group -> dict(rgb uint8 (T+1,3,H,W), state f32 (T+1,26), actions f32 (T,4)) on device."""
    rgb = torch.from_numpy(traj[RGB_KEY][()])                 # (T+1, H, W, 3) uint8
    rgb = rgb.permute(0, 3, 1, 2).contiguous().to(device)     # (T+1, 3, H, W)
    parts = []
    for k in STATE_KEYS:
        v = traj[k][()]
        if v.ndim == 1:
            v = v[:, None]
        parts.append(v.astype(np.float32))
    state = torch.from_numpy(np.concatenate(parts, axis=1)).to(device)   # (T+1, 26)
    actions = torch.from_numpy(traj["actions"][()].astype(np.float32))
    if clip_actions:
        actions = actions.clamp_(-1.0, 1.0)
    actions = actions.to(device)
    assert state.shape[0] == actions.shape[0] + 1 and rgb.shape[0] == state.shape[0], \
        (rgb.shape, state.shape, actions.shape)
    return {"rgb": rgb, "state": state, "actions": actions}


def demo_env_info(demo_path):
    with open(demo_path[:-2] + "json") as f:
        return json.load(f)["env_info"]


class StreamingRGBDemoDataset(Dataset):
    """Slices of (obs_horizon observations, pred_horizon actions), all tensors resident on `device`.

    Same padding convention as the vendored SmallDemoDataset_DiffusionPolicy:
      |o|o|                             observations: obs_horizon
      | |a|a|a|a|a|a|a|a|               actions executed: act_horizon
      |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: pred_horizon
    pad_before = obs_horizon - 1 (first obs repeated), pad_after = pred_horizon - obs_horizon
    (arm stays still: zero xyz delta, gripper copied from the last action).
    """

    def __init__(self, demo_paths, obs_horizon, pred_horizon, device, num_demos=None, clip_actions=True,
                 verbose=True):
        if isinstance(demo_paths, str):
            demo_paths = [demo_paths]
        self.obs_horizon, self.pred_horizon = int(obs_horizon), int(pred_horizon)
        self.device = torch.device(device)
        self.trajectories = []
        self.sources = []
        for path in demo_paths:
            assert os.path.exists(path), f"demo dataset not found: {path}"
            with h5py.File(path, "r") as f:
                keys = _traj_keys(f)
                if num_demos is not None:
                    keys = keys[: int(num_demos)]
                for i, k in enumerate(keys):
                    self.trajectories.append(read_trajectory(f[k], self.device, clip_actions))
                    self.sources.append((path, k))
                    if verbose and (i + 1) % 50 == 0:
                        print(f"[streaming_dataset] {os.path.basename(os.path.dirname(path))}: "
                              f"{i + 1}/{len(keys)} trajectories loaded", flush=True)
        assert self.trajectories, "no trajectories loaded"
        act_dim = self.trajectories[0]["actions"].shape[1]
        self.pad_action_arm = torch.zeros((act_dim - 1,), device=self.device)

        self.slices = []
        total = 0
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon
        for ti, tr in enumerate(self.trajectories):
            L = tr["actions"].shape[0]
            total += L
            self.slices += [(ti, s, s + self.pred_horizon) for s in range(-pad_before, L - self.pred_horizon + pad_after)]
        n_frames = sum(t["rgb"].shape[0] for t in self.trajectories)
        gb = n_frames * int(np.prod(self.trajectories[0]["rgb"].shape[1:])) / 1e9
        if verbose:
            print(f"[streaming_dataset] {len(self.trajectories)} trajectories, {total} transitions, "
                  f"{len(self.slices)} obs sequences, rgb {gb:.2f} GB uint8 on {self.device}", flush=True)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        ti, start, end = self.slices[index]
        tr = self.trajectories[ti]
        L = tr["actions"].shape[0]
        obs_seq = {}
        for k in ("rgb", "state"):
            v = tr[k][max(0, start): start + self.obs_horizon]
            if start < 0:
                v = torch.cat([v[:1].expand(-start, *v.shape[1:]), v], dim=0)
            obs_seq[k] = v
        act = tr["actions"][max(0, start): end]
        if start < 0:
            act = torch.cat([act[:1].expand(-start, act.shape[1]), act], dim=0)
        if end > L:
            pad = torch.cat((self.pad_action_arm, act[-1, -1:]), dim=0)
            act = torch.cat([act, pad[None].expand(end - L, -1)], dim=0)
        assert obs_seq["state"].shape[0] == self.obs_horizon and act.shape[0] == self.pred_horizon
        return {"observations": obs_seq, "actions": act}
