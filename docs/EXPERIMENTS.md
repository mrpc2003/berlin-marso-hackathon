# Experiment log (append-only)

Record every serious run here: command, git commit, checkpoint, eval config + episode count, scores.
Decision rule: easy on 64 episodes, medium/hard on 32 (16-episode evals are noise: ±9 pt on easy).

| date | session | level | exp / ckpt | key settings | eval (cfg, n) | sort_acc | mean_steps | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-13 | S0 | – | – | code changes only (chunked deployment, per-level episode budget, clipped actions, streaming loader, resume, aug) | CPU tests: 10 pass | – | – | no GPU run yet |

## Gates
- Gate 1 (end S1, easy, 64 eps): ≥40% → DP main line; 30–40% → run ACT hedge next; <30% → diagnose + one retrain (Gate 1b), then ACT becomes main line.
- Gate 2 (S3): ACT beats DP by >10 pt on the same 64 episodes → switch.
- Gate 3 (hard, 32 eps): ≥5% include; 0% after two attempts → stop spending on hard.
