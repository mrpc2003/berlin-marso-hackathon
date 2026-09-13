"""Training-only, parameter-free image augmentation for the RGB Diffusion Policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomShiftsAug(nn.Module):
    """DrQ-v2 style random translation: replicate-pad by `pad` px then random-crop back via
    grid_sample. Mimics small camera/object jitter so the policy generalises to the held-out wider
    position randomisation. Colour-safe (no hue change) and parameter-free, so a checkpoint trained
    with it loads into an Agent built without it. Applied by Agent.encode_obs only when training.
    """

    def __init__(self, pad=4):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x):                              # x: (N, C, H, W) float
        n, c, h, w = x.size()
        assert h == w, "RandomShiftsAug expects square images"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
