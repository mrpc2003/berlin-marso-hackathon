import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (REPO, os.path.join(REPO, "il", "baselines", "diffusion_policy")):
    if p not in sys.path:
        sys.path.insert(0, p)
