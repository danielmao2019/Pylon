"""
EvaluationJob progress-related tests.
"""

import json
import os
import tempfile

from agents.manager.evaluation_job import EvaluationJob
from agents.manager.runtime import JobRuntimeParams


def test_evaluationjob_incomplete_and_complete():
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "eval_case")
        config_path = os.path.join(configs_dir, "eval_case.py")

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("config = {}\n")

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = EvaluationJob(
                "python main.py --config-filepath ./configs/eval_case.py"
            )

            # Incomplete
            p0 = job.compute_progress(
                JobRuntimeParams(
                    epochs=1,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )
            assert p0.completed_epochs == 0
            assert p0.progress_percentage == 0.0

            # Complete
            with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
                json.dump(
                    {"aggregated": {"acc": 1.0}, "per_datapoint": {"acc": [1.0]}}, f
                )
            p1 = job.compute_progress(
                JobRuntimeParams(
                    epochs=1,
                    sleep_time=1,
                    outdated_days=30,
                    command_processes={},
                    force_progress_recompute=False,
                )
            )
            assert p1.completed_epochs == 1
            assert p1.progress_percentage == 100.0
        finally:
            os.chdir(cwd)


def test_evaluationjob_progress_file_validation():
    with tempfile.TemporaryDirectory() as temp_root:
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(temp_root, "logs", "eval_case")
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        config_path = os.path.join(configs_dir, "eval_case.py")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(
                "from runners.evaluators.base_evaluator import BaseEvaluator\n"
                "config = {'runner': BaseEvaluator}\n"
            )

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = EvaluationJob(
                "python main.py --config-filepath ./configs/eval_case.py"
            )
            runtime = JobRuntimeParams(
                epochs=1,
                sleep_time=1,
                outdated_days=30,
                command_processes={},
                force_progress_recompute=False,
            )

            # Missing scores file -> incomplete progress
            progress = job.compute_progress(runtime)
            assert progress.completed_epochs == 0
            assert progress.progress_percentage == 0.0

            scores_path = os.path.join(work_dir, "evaluation_scores.json")

            # Empty file still counts as incomplete
            open(scores_path, 'w').close()
            progress = job.compute_progress(runtime)
            assert progress.completed_epochs == 0

            # Malformed JSON should be treated as incomplete
            with open(scores_path, 'w', encoding='utf-8') as f:
                f.write('{invalid-json')
            progress = job.compute_progress(runtime)
            assert progress.completed_epochs == 0

            # Valid JSON marks the evaluation as complete
            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump({"aggregated": {}, "per_datapoint": {}}, f)
            progress = job.compute_progress(runtime)
            assert progress.completed_epochs == 1
            assert progress.progress_percentage == 100.0
        finally:
            os.chdir(cwd)
