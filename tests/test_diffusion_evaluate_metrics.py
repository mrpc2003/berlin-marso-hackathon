import sys
import types

import numpy as np
import pytest
import torch

import diffusion_policy.evaluate as eval_module


def _install_common_stub(monkeypatch):
    mani_skill = types.ModuleType("mani_skill")
    utils = types.ModuleType("mani_skill.utils")
    common = types.ModuleType("mani_skill.utils.common")
    common.to_tensor = lambda obs, device: obs
    utils.common = common
    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill)
    monkeypatch.setitem(sys.modules, "mani_skill.utils", utils)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.common", common)


class _FakeAgent:
    def __init__(self, num_envs=8, horizon=2, training=True):
        self.num_envs = num_envs
        self.horizon = horizon
        self.training = training
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1

    def train(self):
        self.training = True
        self.train_calls += 1

    def get_action(self, obs):
        return torch.zeros((self.num_envs, self.horizon, 4), dtype=torch.float32)


class _FakeEvalEnv:
    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.batch = 0

    def reset(self):
        return torch.zeros((self.num_envs, 1)), {}

    def step(self, action):
        start = self.batch * self.num_envs
        values = torch.arange(start, start + self.num_envs, dtype=torch.float32)
        self.batch += 1
        info = {
            "final_info": {
                "episode": {
                    "success_at_end": values,
                    "success_once": values + 100,
                },
                "sort_accuracy": values / 100,
            }
        }
        return torch.zeros((self.num_envs, 1)), torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), torch.ones(self.num_envs, dtype=torch.bool), info


class _FailingEvalEnv(_FakeEvalEnv):
    def step(self, action):
        raise RuntimeError("step failed")


class _FakePbar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.updates = []
        self.closed = False
        self.instances.append(self)

    def update(self, n):
        self.updates.append(n)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("n", [8, 32, 10])
def test_evaluate_returns_flat_episode_metrics_for_requested_count(monkeypatch, n):
    _install_common_stub(monkeypatch)
    metrics = eval_module.evaluate(
        n,
        _FakeAgent(num_envs=8),
        _FakeEvalEnv(num_envs=8),
        torch.device("cpu"),
        "physx_cpu",
        progress_bar=False,
    )

    assert metrics["success_at_end"].shape == (n,)
    assert metrics["success_once"].shape == (n,)
    assert metrics["sort_accuracy"].shape == (n,)
    np.testing.assert_array_equal(metrics["success_at_end"], np.arange(n, dtype=np.float32))
    np.testing.assert_array_equal(metrics["success_once"], np.arange(n, dtype=np.float32) + 100)


def test_evaluate_closes_progress_bar_on_success(monkeypatch):
    _install_common_stub(monkeypatch)
    _FakePbar.instances.clear()
    monkeypatch.setattr(eval_module, "tqdm", _FakePbar)

    eval_module.evaluate(
        10,
        _FakeAgent(num_envs=8),
        _FakeEvalEnv(num_envs=8),
        torch.device("cpu"),
        "gpu",
        progress_bar=True,
    )

    assert len(_FakePbar.instances) == 1
    assert _FakePbar.instances[0].updates == [8, 2]
    assert _FakePbar.instances[0].closed


def test_evaluate_closes_progress_bar_and_preserves_eval_mode_on_failure(monkeypatch):
    _install_common_stub(monkeypatch)
    _FakePbar.instances.clear()
    monkeypatch.setattr(eval_module, "tqdm", _FakePbar)
    agent = _FakeAgent(num_envs=8, training=False)

    with pytest.raises(RuntimeError, match="step failed"):
        eval_module.evaluate(
            8,
            agent,
            _FailingEvalEnv(num_envs=8),
            torch.device("cpu"),
            "gpu",
            progress_bar=True,
        )

    assert len(_FakePbar.instances) == 1
    assert _FakePbar.instances[0].closed
    assert agent.training is False
    assert agent.train_calls == 0
