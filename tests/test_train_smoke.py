"""End-to-end CPU smoke: train 6 iters (no simulator) -> latest.pt -> resume 3 more -> strip -> load via
warehouse_sort.il_policy.load_dp_rgb -> act on fake observations."""
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
H5 = os.path.join(REPO, "il", "demos", "easy", "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5")
TMP = os.path.join(REPO, "tests", "_tmp")
pytestmark = pytest.mark.skipif(not os.path.exists(H5), reason="easy rgb demos not downloaded")

COMMON = ["--demo-path", H5, "--env-id", "WarehouseSort-v1", "--control-mode", "pd_ee_delta_pos",
          "--sim-backend", "gpu", "--max-episode-steps", "250", "--eval-freq", "0", "--no-cuda",
          "--num-demos", "2", "--batch-size", "4", "--visual-encoder", "plain_conv", "--unet-dims", "32", "64",
          "--save-freq", "3", "--log-freq", "1", "--no-amp", "--image-aug-pad", "4", "--proprio-noise-std", "0.01"]


def _run(extra):
    cmd = [sys.executable, "train_rgbd.py"] + COMMON + extra
    r = subprocess.run(cmd, cwd=os.path.join(REPO, "il", "baselines", "diffusion_policy"), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_train_resume_strip_load():
    ckpt_dir = os.path.join(TMP, "ckpt")
    _run(["--total-iters", "6", "--exp-name", "smoke", "--ckpt-dir", ckpt_dir])
    latest = os.path.join(ckpt_dir, "latest.pt")
    assert os.path.exists(latest) and os.path.exists(os.path.join(ckpt_dir, "final.submit.pt"))
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    assert ck["iteration"] == 6 and ck["config"]["visual_encoder"] == "plain_conv" and ck["config"]["clip_actions"]
    out = _run(["--total-iters", "9", "--exp-name", "smoke", "--ckpt-dir", ckpt_dir, "--resume", latest])
    assert "resumed" in out and "iteration 7/9" in out
    ck2 = torch.load(latest, map_location="cpu", weights_only=False)
    assert ck2["iteration"] == 9

    stripped = os.path.join(TMP, "smoke.pt")
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "strip_ckpt.py"), latest, stripped, "--set", "act_horizon=4"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    st = torch.load(stripped, map_location="cpu", weights_only=False)
    assert set(st) >= {"model", "config"} and st["config"]["act_horizon"] == 4

    sys.path.insert(0, REPO)
    from warehouse_sort.il_policy import load_dp_rgb
    import gymnasium.spaces as spaces
    import numpy as np
    sample = {"rgb": torch.zeros(2, 128, 128, 3, dtype=torch.uint8), "state": torch.zeros(2, 26)}
    pol = load_dp_rgb(stripped, sample, spaces.Box(-1, 1, (4,), np.float32), "cpu")
    assert pol.act_horizon == 4 and pol.obs_horizon == 2
    a = pol.act(sample)
    assert a.shape == (2, 4) and a.abs().max() <= 1.0 and len(pol._queue) == 3
    # deterministic given the seeded generator
    pol2 = load_dp_rgb(stripped, sample, spaces.Box(-1, 1, (4,), np.float32), "cpu")
    assert torch.allclose(a, pol2.act(sample))
