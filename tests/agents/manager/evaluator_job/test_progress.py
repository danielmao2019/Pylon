"""
EvaluationJob progress-related tests.
"""
import json
import os
import tempfile

from agents.manager.evaluation_job import EvaluationJob


def test_evaluationjob_incomplete_and_complete():
    with tempfile.TemporaryDirectory() as work_dir:
        # Incomplete
        p0 = EvaluationJob.get_progress(work_dir, None)
        assert p0.completed_epochs == 0
        assert p0.progress_percentage == 0.0

        # Complete
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {"acc": 1.0}, "per_datapoint": {"acc": [1.0]}}, f)
        p1 = EvaluationJob.get_progress(work_dir, None)
        assert p1.completed_epochs == 1
        assert p1.progress_percentage == 100.0


def test_evaluationjob_check_files_exist():
    with tempfile.TemporaryDirectory() as work_dir:
        assert EvaluationJob._check_files_exist(work_dir) is False
        # empty file
        open(os.path.join(work_dir, "evaluation_scores.json"), 'w').close()
        assert EvaluationJob._check_files_exist(work_dir) is False
        # valid json
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump({"aggregated": {}, "per_datapoint": {}}, f)
        assert EvaluationJob._check_files_exist(work_dir) is True
