"""Diagnose WHY a policy fails: per-step log of gripper command, grasp state, TCP height and the
TCP-to-target-parcel xy error, plus per-episode phase summaries. Uses privileged env state for
ANALYSIS ONLY (the policy still acts from the observation) -- never part of a submission.

  python tools/rollout_logger.py difficulty=easy obs_mode=rgb policy=warehouse_sort.il_policy:load_dp_rgb \\
      checkpoint=<ckpt> eval_config=conf/eval/eval32.yaml num_envs=8 +log_episodes=8 +log_out=outputs/diag.json

Prints, per episode: parcels sorted, steps to first grasp, number of gripper close->open flickers,
longest run of consecutive "close" commands before the first lift, mean xy error at the moment the
gripper first closes, and whether the arm ever got within 2 cm of each parcel.
"""

import json
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from warehouse_sort.utils import load_agent, log_run_header, make_env, to_device


def _policy_kwargs_to_container(policy_kwargs):
    if policy_kwargs is None:
        return {}
    if OmegaConf.is_config(policy_kwargs):
        return OmegaConf.to_container(policy_kwargs, resolve=True) or {}
    return dict(policy_kwargs)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    assert cfg.checkpoint and cfg.policy
    log_run_header(cfg, "diag")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    eval_cfg = OmegaConf.load(cfg.eval_config) if cfg.get("eval_config") else None
    seeds = list(eval_cfg.eval.seeds) if eval_cfg else [5000 + i for i in range(cfg.num_envs)]
    randomization = (eval_cfg.get("randomization", None) if eval_cfg else None) or cfg.randomization
    n_eps = int(cfg.get("log_episodes", cfg.num_envs))
    n_envs = min(cfg.num_envs, n_eps)
    env, _ = make_env(cfg, cfg.obs_mode, randomization, num_envs=n_envs)
    base = env.unwrapped
    policy_kwargs = _policy_kwargs_to_container(cfg.get("policy_kwargs"))
    agent, _ = load_agent(cfg.checkpoint, env, device, entrypoint=cfg.policy, policy_kwargs=policy_kwargs)

    obs, _ = env.reset(seed=[int(s) for s in seeds[:n_envs]])
    if hasattr(agent, "reset"):
        agent.reset()
    obs = to_device(obs, device)
    T = int(cfg.max_episode_steps)
    P = base.num_parcels
    grip_cmd = np.zeros((T, n_envs), np.float32)
    grasped = np.zeros((T, n_envs), bool)
    tcp = np.zeros((T, n_envs, 3), np.float32)
    parcel_xy = np.zeros((T, n_envs, P, 2), np.float32)
    parcel_z = np.zeros((T, n_envs, P), np.float32)
    sorted_cnt = np.zeros((T, n_envs), np.float32)
    for t in range(T - 1):
        a = agent.act(obs, deterministic=True)
        obs, _, _, _, info = env.step(a)
        obs = to_device(obs, device)
        grip_cmd[t] = a[:, 3].detach().cpu().numpy()
        grasped[t] = info["is_grasped"].detach().cpu().numpy() if "is_grasped" in info else base.evaluate()["is_grasped"].cpu().numpy()
        tcp[t] = base.agent.tcp_pose.p.detach().cpu().numpy()
        for j, parcel in enumerate(base.parcels):
            pp = parcel.pose.p.detach().cpu().numpy()
            parcel_xy[t, :, j] = pp[:, :2]
            parcel_z[t, :, j] = pp[:, 2]
        sorted_cnt[t] = base.evaluate()["success_count"].cpu().numpy()

    report = []
    for e in range(n_envs):
        g = grip_cmd[:T - 1, e]
        closed = g < 0
        first_close = int(np.argmax(closed)) if closed.any() else -1
        first_grasp = int(np.argmax(grasped[:T - 1, e])) if grasped[:T - 1, e].any() else -1
        flips = int(np.sum(np.abs(np.diff(closed.astype(int)))))
        # longest run of consecutive close commands
        best = run = 0
        for c in closed:
            run = run + 1 if c else 0
            best = max(best, run)
        d = np.linalg.norm(tcp[:T - 1, e, None, :2] - parcel_xy[:T - 1, e], axis=-1)   # (T, P)
        min_xy_err = d.min(axis=0).tolist()
        xy_err_at_close = float(d[first_close].min()) if first_close >= 0 else None
        report.append(dict(
            env=e, seed=int(seeds[e]), sorted=float(sorted_cnt[T - 2, e]), parcels=P,
            first_close_step=first_close, first_grasp_step=first_grasp, gripper_flips=flips,
            longest_close_run=best, xy_err_at_first_close=xy_err_at_close,
            min_xy_err_per_parcel=[round(x, 4) for x in min_xy_err],
            reached_within_2cm=[bool(x < 0.02) for x in min_xy_err],
            n_plans=getattr(agent, "n_plans", None),
        ))
        print(json.dumps(report[-1]))
    print(f"mean sorted {np.mean([r['sorted'] for r in report]):.2f}/{P}  "
          f"grasped-at-least-once {np.mean([r['first_grasp_step'] >= 0 for r in report]):.2f}")
    out = cfg.get("log_out")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(checkpoint=cfg.checkpoint, level=cfg.difficulty.name, episodes=report), f, indent=1)
        np.savez_compressed(out.replace(".json", "_traces.npz"), grip_cmd=grip_cmd, grasped=grasped, tcp=tcp,
                            parcel_xy=parcel_xy, parcel_z=parcel_z, sorted_cnt=sorted_cnt)
        print("wrote", out)
    env.close()


if __name__ == "__main__":
    main()
