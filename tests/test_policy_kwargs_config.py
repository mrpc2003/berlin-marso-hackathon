import importlib
import sys
import types

from omegaconf import OmegaConf


def _stub_mani_skill_imports(monkeypatch):
    mani_skill = types.ModuleType("mani_skill")
    utils = types.ModuleType("mani_skill.utils")
    wrappers = types.ModuleType("mani_skill.utils.wrappers")
    flatten = types.ModuleType("mani_skill.utils.wrappers.flatten")
    record = types.ModuleType("mani_skill.utils.wrappers.record")
    vector = types.ModuleType("mani_skill.vector")
    vector_wrappers = types.ModuleType("mani_skill.vector.wrappers")
    vector_gym = types.ModuleType("mani_skill.vector.wrappers.gymnasium")

    flatten.FlattenRGBDObservationWrapper = object
    record.RecordEpisode = object
    vector_gym.ManiSkillVectorEnv = object

    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill)
    monkeypatch.setitem(sys.modules, "mani_skill.utils", utils)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.wrappers", wrappers)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.wrappers.flatten", flatten)
    monkeypatch.setitem(sys.modules, "mani_skill.utils.wrappers.record", record)
    monkeypatch.setitem(sys.modules, "mani_skill.vector", vector)
    monkeypatch.setitem(sys.modules, "mani_skill.vector.wrappers", vector_wrappers)
    monkeypatch.setitem(sys.modules, "mani_skill.vector.wrappers.gymnasium", vector_gym)


def _fresh_import(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_policy_kwargs_helpers_accept_empty_plain_dict_and_missing(monkeypatch):
    _stub_mani_skill_imports(monkeypatch)
    for module_name in ("eval", "tools.rollout_logger"):
        module = _fresh_import(module_name)
        assert module._policy_kwargs_to_container(None) == {}
        assert module._policy_kwargs_to_container({}) == {}
        assert module._policy_kwargs_to_container({"act_horizon": 4}) == {"act_horizon": 4}


def test_policy_kwargs_helpers_resolve_nonempty_dictconfig(monkeypatch):
    _stub_mani_skill_imports(monkeypatch)
    cfg = OmegaConf.create({"steps": 32, "policy_kwargs": {"num_inference_steps": "${steps}"}})
    for module_name in ("eval", "tools.rollout_logger"):
        module = _fresh_import(module_name)
        assert module._policy_kwargs_to_container(OmegaConf.create({})) == {}
        assert module._policy_kwargs_to_container(cfg.policy_kwargs) == {"num_inference_steps": 32}


def test_eval_main_writes_actual_and_requested_episode_counts(monkeypatch, tmp_path):
    _stub_mani_skill_imports(monkeypatch)
    module = _fresh_import("eval")
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text("eval:\n  n_episodes: 8\n  seeds: [5000]\n")
    cfg = OmegaConf.create({"checkpoint": "own.pt", "eval_config": str(eval_path),
        "device": "cpu", "num_envs": 8, "obs_mode": "rgb", "policy": "example:load",
        "policy_kwargs": {}, "randomization": {}, "max_episode_steps": 250,
        "difficulty": {"name": "easy"}, "results_file": str(tmp_path / "results.jsonl"),
        "record_video": False})
    env = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(module, "log_run_header", lambda *args: None)
    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: (env, None))
    monkeypatch.setattr(module, "load_agent", lambda *args, **kwargs: (object(), None))
    monkeypatch.setattr(module, "rollout_metrics", lambda *args: {"n_episodes": 8, "sort_accuracy": 0.25})
    monkeypatch.setattr(module, "print_metrics", lambda *args, **kwargs: None)
    records = []
    monkeypatch.setattr(module, "append_jsonl", lambda path, row: records.append(row))
    module.main.__wrapped__(cfg)
    assert len(records) == 1
    assert records[0]["n_episodes"] == 8
    assert records[0]["requested_n_episodes"] == 8
    assert records[0]["sort_accuracy"] == 0.25

