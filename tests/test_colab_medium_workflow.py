import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

import tools.make_colab_medium_notebook as gen
import tools.run_colab_medium as supervisor


def test_notebook_contract_and_default_off():
    nb = gen.build_notebook()
    gen.validate_notebook(nb)
    cells = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    assert cells["drive-mount"] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert "time.strftime('%Y%m%d-%H%M%S')" in cells["setup-py312"]
    assert "RUN_MAIN = False" in cells["launch-medium"]
    assert "start_new_session=True" in cells["launch-medium"]
    assert "child.poll() is None" in cells["launch-medium"]
    assert "uv','venv','--seed','--python','3.12.3'" in cells["setup-py312"]
    assert "torch==2.11.0" in cells["setup-py312"] and "torchvision==0.26.0" in cells["setup-py312"]
    assert "mani-skill==3.0.1" in cells["setup-py312"] and "diffusers==0.38.0" in cells["setup-py312"]
    assert "KAGGLE" not in json.dumps(nb) and "GH_TOKEN" not in json.dumps(nb)


def test_real_tqdm_loss_parser_and_nonfinite_rejection():
    import re, math
    node=next(n for n in ast.parse(gen.SMOKE).body if isinstance(n,ast.FunctionDef) and n.name=='losses')
    scope={'re':re,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'loss-parser','exec'),scope)
    assert scope['losses']('300/300 [5.23it/s, loss=0.1234]') == [0.1234]
    with pytest.raises(AssertionError): scope['losses']('loss=nan')


def test_medium_level_32_episode_500_step_contract():
    nb_text = json.dumps(gen.build_notebook())
    runner = Path(supervisor.__file__).read_text()
    assert "difficulty=medium" in runner
    assert '"flags.num_eval_episodes=32"' in runner
    assert '"flags.eval_freq=5000"' in runner
    assert '"flags.save_freq=1000"' in runner
    assert '"flags.total_iters=30000"' in runner
    assert 'int(plan["max_episode_steps"]) == 500' in runner
    assert '["requested_n_episodes"] == 32' in runner
    assert '["max_episode_steps"] == 500' in runner
    assert "DIFFICULTY_KWARGS['medium']" in nb_text
    assert "scripted_sorted':sorted_count" in nb_text


def test_stage_probe_validates_variable_t_schema(tmp_path, capsys):
    h5 = tmp_path / "trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5"
    with h5py.File(h5, "w") as f:
        for i, t in enumerate((3, 5)):
            g = f.create_group(f"traj_{i}")
            g.create_dataset("actions", data=np.zeros((t, 4), np.float32))
            g.create_dataset("obs/agent/qpos", data=np.zeros((t + 1, 9), np.float32))
            g.create_dataset("obs/agent/qvel", data=np.zeros((t + 1, 9), np.float32))
            g.create_dataset("obs/extra/tcp_pose", data=np.zeros((t + 1, 7), np.float32))
            g.create_dataset("obs/extra/is_grasped", data=np.zeros((t + 1, 1), np.float32))
            g.create_dataset("obs/sensor_data/scene_camera/rgb", data=np.zeros((t + 1, 128, 128, 3), np.uint8))
    h5.with_suffix(".json").write_text(json.dumps({"env_info":{"env_id":"WarehouseSort-v1","env_kwargs":{"num_parcels":4,"fixed_poses":False}},"episodes": [{"episode_id": 0}]}))
    report = tmp_path / "report.json"
    src = gen.DATA_PROBE.replace("assert len(keys) == 200, len(keys)", "assert len(keys) == 2, len(keys)")
    env = {"MARSO_H5": str(h5), "MARSO_DATA_REPORT": str(report)}
    old = os.environ.copy()
    os.environ.update(env)
    try:
        exec(compile(src, "<DATA_PROBE>", "exec"), {})
    finally:
        os.environ.clear()
        os.environ.update(old)
    data = json.loads(report.read_text())
    assert data["lengths"] == [3, 5]
    assert data["metadata_episodecount_mismatch"] is True
    assert "WARNING metadata episode count mismatch" in capsys.readouterr().out


def test_run_stage_preserves_failure_log(tmp_path):
    log = tmp_path / "fail.log"
    updates = []
    with pytest.raises(RuntimeError, match=str(log)):
        supervisor.run_stage(
            [sys.executable, "-c", "print('kept failure log'); raise SystemExit(7)"],
            log,
            30,
            lambda **kw: updates.append(kw),
            os.environ.copy(),
            tmp_path,
        )
    assert "kept failure log" in log.read_text()
    assert updates and updates[0]["log"] == str(log)


def test_no_easy_target_or_old_run_dirs_in_new_workflow():
    text = Path(supervisor.__file__).read_text() + "\n" + Path(gen.__file__).read_text()
    forbidden = ("dp_rgb_easy", "difficulty=easy", "ckpts/easy", "easyS1", "flags.resume")
    for token in forbidden:
        assert token not in text
    assert "runs/medium" in text and "ckpts/medium" in text
    ast.parse(Path(supervisor.__file__).read_text())
    ast.parse(Path(gen.__file__).read_text())
