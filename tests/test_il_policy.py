"""CPU tests for the deployment executor (no simulator needed)."""
import numpy as np
import torch

from warehouse_sort.constants import START_QPOS
from warehouse_sort.il_policy import ChunkedPolicy, resolve_policy_config


class _CountingPolicy(ChunkedPolicy):
    """Plans a chunk whose values encode (plan index, step index) so we can check execution order."""

    def __init__(self, obs_horizon=2, act_horizon=4, **kw):
        super().__init__(obs_horizon, act_horizon, "cpu", **kw)
        self.seen_hist_shapes = []

    def _prep(self, obs):
        return {"state": obs["state"].float(), "rgb": obs["rgb"]}

    def _plan(self, obs_seq):
        self.seen_hist_shapes.append(tuple(obs_seq["rgb"].shape))
        B = obs_seq["state"].shape[0]
        idx = torch.arange(self.act_horizon).float().view(1, -1, 1)
        chunk = torch.zeros(B, self.act_horizon, 4) + self.n_plans * 0.1 + idx * 0.01   # stays inside the [-1,1] clamp
        chunk[:, :, 3] = 0.3 if self.n_plans % 2 == 0 else -0.3
        return chunk


def _obs(B=3, qpos7=None, t=0):
    state = torch.zeros(B, 26)
    state[:, :9] = torch.tensor(START_QPOS)
    if qpos7 is not None:
        state[:, :7] = torch.as_tensor(qpos7)
    rgb = torch.full((B, 128, 128, 3), t, dtype=torch.uint8)
    return {"state": state, "rgb": rgb}


def test_chunk_is_executed_in_order_then_replanned():
    pol = _CountingPolicy(obs_horizon=2, act_horizon=4)
    q = torch.tensor(START_QPOS[:7])
    acts = []
    for t in range(9):
        q = q + 0.05                      # arm moves away from home each step (no reset trigger)
        acts.append(pol.act(_obs(qpos7=q, t=t))[0, 0].item())
    assert pol.n_plans == 3, pol.n_plans
    assert np.allclose(acts[:4], [0.00, 0.01, 0.02, 0.03], atol=1e-6)
    assert np.allclose(acts[4:8], [0.10, 0.11, 0.12, 0.13], atol=1e-6)
    assert np.isclose(acts[8], 0.20, atol=1e-6)


def test_history_is_rolling_not_duplicated():
    pol = _CountingPolicy(obs_horizon=3, act_horizon=1)
    q = torch.tensor(START_QPOS[:7])
    for t in range(4):
        q = q + 0.05
        pol.act(_obs(qpos7=q, t=t))
    # shapes: (B, obs_horizon, H, W, 3)
    assert all(s == (3, 3, 128, 128, 3) for s in pol.seen_hist_shapes)
    hist = pol._stack()["rgb"][0, :, 0, 0, 0].tolist()
    assert hist == [1, 2, 3], hist          # last three frames, oldest first


def test_reset_detected_when_arm_returns_home_or_teleports():
    pol = _CountingPolicy(obs_horizon=2, act_horizon=8)
    q = torch.tensor(START_QPOS[:7])
    pol.act(_obs(qpos7=q))                  # plan 0 at home
    pol.act(_obs(qpos7=q + 0.2))            # left home, still executing plan 0
    assert pol.n_plans == 1 and len(pol._queue) == 6
    pol.act(_obs(qpos7=q))                  # back at exact home -> new episode -> replan
    assert pol.n_plans == 2 and len(pol._queue) == 7
    pol.act(_obs(qpos7=q + 0.05))
    pol.act(_obs(qpos7=q + 0.9))            # teleport (jump > 0.3) -> replan
    assert pol.n_plans == 3
    pol.reset()
    assert pol._hist is None and pol._queue == []


def test_batch_size_change_resets():
    pol = _CountingPolicy(obs_horizon=2, act_horizon=8)
    pol.act(_obs(B=3))
    pol.act(_obs(B=5, qpos7=torch.tensor(START_QPOS[:7]) + 0.1))
    assert pol.n_plans == 2


def test_gripper_binarize():
    pol = _CountingPolicy(obs_horizon=2, act_horizon=2, gripper_binarize=True)
    a = pol.act(_obs())
    assert torch.all(a[:, 3] == 1.0)        # plan 0 -> +0.3 -> +1
    pol.act(_obs(qpos7=torch.tensor(START_QPOS[:7]) + 0.1))
    a = pol.act(_obs(qpos7=torch.tensor(START_QPOS[:7]) + 0.2))
    assert torch.all(a[:, 3] == -1.0)       # plan 1 -> -0.3 -> -1


def test_resolve_policy_config_precedence():
    ckpt = {"config": {"act_horizon": 4, "num_inference_steps": 32, "unet_dims": (64, 128, 256)}}
    cfg = resolve_policy_config(ckpt, {"num_inference_steps": 8, "act_horizon": None})
    assert cfg["act_horizon"] == 4 and cfg["num_inference_steps"] == 8
    assert cfg["obs_horizon"] == 2 and cfg["unet_dims"] == [64, 128, 256]
