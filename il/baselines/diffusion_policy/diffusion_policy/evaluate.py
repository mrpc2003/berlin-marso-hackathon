from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm


def _episode_vector(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value).reshape(-1)


def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    from mani_skill.utils import common  # lazy: keeps this module importable without the simulator

    was_training = getattr(agent, "training", None)
    agent.eval()
    pbar = tqdm(total=n) if progress_bar else None
    try:
        with torch.no_grad():
            eval_metrics = defaultdict(list)
            obs, info = eval_envs.reset()
            eps_count = 0
            while eps_count < n:
                obs = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs)
                if sim_backend == "physx_cpu":
                    action_seq = action_seq.cpu().numpy()
                for i in range(action_seq.shape[1]):
                    obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                    if truncated.any():
                        break

                if truncated.any():
                    assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                    if isinstance(info["final_info"], dict):
                        for k, v in info["final_info"]["episode"].items():
                            eval_metrics[k].append(_episode_vector(v))
                        # also log sort_accuracy (fraction of parcels sorted by the time limit) -- a
                        # partial-credit task-progress metric, better than full-episode success alone.
                        if "sort_accuracy" in info["final_info"]:
                            eval_metrics["sort_accuracy"].append(_episode_vector(info["final_info"]["sort_accuracy"]))
                    else:
                        for final_info in info["final_info"]:
                            for k, v in final_info["episode"].items():
                                eval_metrics[k].append(_episode_vector(v))
                            if "sort_accuracy" in final_info:
                                eval_metrics["sort_accuracy"].append(_episode_vector(final_info["sort_accuracy"]))
                    take = min(eval_envs.num_envs, n - eps_count)
                    eps_count += take
                    if pbar is not None:
                        pbar.update(take)
        for k, values in eval_metrics.items():
            eval_metrics[k] = np.concatenate(values, axis=0)[:n]
        return eval_metrics
    finally:
        if pbar is not None:
            pbar.close()
        if was_training is False:
            agent.eval()
        else:
            agent.train()
