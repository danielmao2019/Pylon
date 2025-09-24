"""
Integration test for Manager.build_jobs with lightweight setup.
"""
import os
import tempfile
import json

from agents.manager.manager import Manager
from agents.manager.base_job import BaseJob
from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob


class DummyGPU:
    def __init__(self, processes):
        self.processes = processes


class DummySystemMonitor:
    def __init__(self, connected_gpus):
        self.connected_gpus = connected_gpus


class DummyProcessInfo:
    def __init__(self, cmd):
        self.cmd = cmd
    def to_dict(self):  # for serialization in BaseJob._serialize
        return {"cmd": self.cmd}


def test_build_jobs_minimal_integration(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_root:
        # Layout: ./configs/x.py -> ./logs/x
        configs_dir = os.path.join(temp_root, "configs")
        logs_dir = os.path.join(temp_root, "logs")
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Create two configs
        cfg_train = os.path.join(configs_dir, "train_exp.py")
        cfg_eval = os.path.join(configs_dir, "eval_exp.py")

        # Minimal config contents with fields used by code
        with open(cfg_train, 'w') as f:
            f.write("config = { 'epochs': 100 }\n")
        with open(cfg_eval, 'w') as f:
            f.write("config = { }\n")

        # Create matching logs dirs
        work_train = os.path.join(logs_dir, "train_exp")
        work_eval = os.path.join(logs_dir, "eval_exp")
        os.makedirs(work_train, exist_ok=True)
        os.makedirs(work_eval, exist_ok=True)

        # Trainer pattern (epoch_0 with validation_scores.json etc.)
        epoch0 = os.path.join(work_train, "epoch_0")
        os.makedirs(epoch0, exist_ok=True)
        with open(os.path.join(epoch0, "validation_scores.json"), 'w') as f:
            json.dump({"acc": 0.9}, f)
        with open(os.path.join(epoch0, "optimizer_buffer.json"), 'w') as f:
            json.dump({"lr": 1e-3}, f)
        # minimal torch file skipped (calculate_progress will count 1 epoch without loading)
        with open(os.path.join(epoch0, "training_losses.pt"), 'wb') as f:
            f.write(b"\x00")

        # Evaluator pattern
        with open(os.path.join(work_eval, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 1.0}, "per_datapoint": {"acc": [1.0]}}, f)

        # Point CWD to temp_root so BaseJob path helpers resolve
        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            # Prepare dummy system monitors (empty; not critical for detection)
            monitors = {"dummy": DummySystemMonitor(connected_gpus=[DummyGPU(processes=[
                DummyProcessInfo(cmd="python main.py --config-filepath ./configs/train_exp.py")
            ])])}

            m = Manager(config_files=["./configs/train_exp.py", "./configs/eval_exp.py"], epochs=1, system_monitors=monitors)
            jobs = m.build_jobs()

            assert set(jobs.keys()) == {"./configs/train_exp.py", "./configs/eval_exp.py"}
            # Training job
            tjob = jobs["./configs/train_exp.py"]
            assert isinstance(tjob, BaseJob)
            assert isinstance(tjob, TrainingJob)
            assert tjob.work_dir == "./logs/train_exp"
            assert tjob.progress is not None
            # Evaluator job
            ejob = jobs["./configs/eval_exp.py"]
            assert isinstance(ejob, BaseJob)
            assert isinstance(ejob, EvaluationJob)
            assert ejob.work_dir == "./logs/eval_exp"
            assert ejob.progress is not None
        finally:
            os.chdir(cwd)

