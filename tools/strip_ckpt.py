"""Strip a trainer checkpoint down to what the judge needs: EMA weights + policy `config`.

Full trainer checkpoints also carry the raw agent, optimizer, LR scheduler, EMA bookkeeping and
RNG state (needed for --resume) and are ~4x larger than the weights alone. The submission
checkpoint keeps only:
    {"model": <ema state_dict>, "config": <policy config>, "iteration": int, "source": str}
which `warehouse_sort.il_policy.load_dp{,_rgb}` accepts (it looks for ema_agent / model / agent).

  python tools/strip_ckpt.py il/baselines/diffusion_policy/runs/<exp>/checkpoints/best_eval_sort_accuracy.pt \
      checkpoints/rgb_dp_easy.pt [--fp16]
"""

import argparse
import os

import torch


def strip(src, dst, fp16=False, config_overrides=None):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_agent") or ckpt.get("model") or ckpt.get("agent")
    assert sd is not None, f"no weights in {src} (keys={list(ckpt)})"
    if fp16:
        sd = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}
    cfg = dict(ckpt.get("config") or {})
    cfg.update(config_overrides or {})
    out = {"model": sd, "config": cfg, "iteration": ckpt.get("iteration"), "source": os.path.abspath(src)}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    mb = os.path.getsize(dst) / 1e6
    print(f"{src} -> {dst}  ({mb:.1f} MB, fp16={fp16}, config keys={sorted(cfg)})")
    if mb > 95:
        print("WARNING: > 95 MB -- GitHub rejects files over 100 MB; use --fp16 or a Release asset.")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fp16", action="store_true", help="store weights in fp16 (halves size)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override policy config entries, e.g. act_horizon=4 num_inference_steps=32 gripper_binarize=true")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        vl = v.lower()
        ov[k] = True if vl == "true" else False if vl == "false" else (int(v) if v.lstrip("-").isdigit() else (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v))
    strip(a.src, a.dst, a.fp16, ov)
