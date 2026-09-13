import os

import h5py
import numpy as np
import pytest
import torch

from diffusion_policy.streaming_dataset import StreamingRGBDemoDataset, read_trajectory

H5 = os.path.join(os.path.dirname(__file__), "..", "il", "demos", "easy", "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
pytestmark = pytest.mark.skipif(not os.path.exists(H5), reason="easy rgb demos not downloaded")


def test_read_trajectory_shapes_and_clipping():
    with h5py.File(H5, "r") as f:
        raw = read_trajectory(f["traj_0"], "cpu", clip_actions=False)
        clipped = read_trajectory(f["traj_0"], "cpu", clip_actions=True)
    T = clipped["actions"].shape[0]
    assert clipped["rgb"].shape == (T + 1, 3, 128, 128) and clipped["rgb"].dtype == torch.uint8
    assert clipped["state"].shape == (T + 1, 26) and clipped["state"].dtype == torch.float32
    assert raw["actions"][:, :3].abs().max() > 1.0, "demo actions are known to exceed [-1,1]"
    assert clipped["actions"].abs().max() <= 1.0
    assert torch.equal(clipped["actions"][:, 3], raw["actions"][:, 3])   # gripper already +-1
    # proprio layout: qpos(9) first; joint 7/8 are the fingers, opened at 0.04 at the start
    assert np.allclose(clipped["state"][0, 7:9].numpy(), [0.04, 0.04], atol=1e-3)
    assert clipped["state"][0, 25] == 0.0     # is_grasped false at t=0


def test_dataset_slices_and_padding():
    ds = StreamingRGBDemoDataset(H5, obs_horizon=2, pred_horizon=16, device="cpu", num_demos=3, verbose=False)
    L = ds.trajectories[0]["actions"].shape[0]
    per_traj = (L - 16 + 14) - (-1)          # range(-pad_before, L - pred + pad_after)
    assert len(ds) == 3 * per_traj
    first = ds[0]                              # start = -1 -> first obs/action repeated once
    assert first["observations"]["rgb"].shape == (2, 3, 128, 128)
    assert first["observations"]["state"].shape == (2, 26) and first["actions"].shape == (16, 4)
    assert torch.equal(first["observations"]["state"][0], first["observations"]["state"][1])
    last = ds[per_traj - 1]                    # end > L -> padded with zero xyz + last gripper
    assert torch.all(last["actions"][-1, :3] == 0) and last["actions"][-1, 3] == ds.trajectories[0]["actions"][-1, 3]


def test_multiple_files_concatenate():
    ds = StreamingRGBDemoDataset([H5, H5], obs_horizon=2, pred_horizon=16, device="cpu", num_demos=2, verbose=False)
    assert len(ds.trajectories) == 4 and len(ds.sources) == 4
