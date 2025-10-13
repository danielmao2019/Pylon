import os
import tempfile

from agents.manager.training_job import TrainingJob


def test_base_job_get_log_last_update(create_real_config):
    """Test log timestamp detection via TrainingJob implementation."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "log_case")
        config_path = os.path.join(configs_dir, "log_case.py")

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        create_real_config(
            config_path, work_dir, epochs=100, early_stopping_enabled=False
        )

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = TrainingJob("python main.py --config-filepath ./configs/log_case.py")

            # Ensure no logs exist yet
            assert job.get_log_last_update() is None

            log_file = os.path.join(job.work_dir, "train_val_latest.log")
            with open(log_file, 'w', encoding='utf-8') as handle:
                handle.write("log content")

            timestamp = job.get_log_last_update()
            assert timestamp is not None
            assert isinstance(timestamp, float)
        finally:
            os.chdir(cwd)


def test_base_job_get_artifact_last_update(
    EXPECTED_FILES, create_epoch_files, create_real_config
):
    """Test artifact timestamp detection for TrainingJob."""
    with tempfile.TemporaryDirectory() as temp_root:
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "artifact_case")
        config_path = os.path.join(configs_dir, "artifact_case.py")

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(configs_dir, exist_ok=True)

        create_real_config(
            config_path, work_dir, epochs=100, early_stopping_enabled=False
        )

        cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            job = TrainingJob(
                "python main.py --config-filepath ./configs/artifact_case.py"
            )

            # Initially no artifacts
            assert job.get_artifact_last_update() is None

            # Create epoch 0 files
            create_epoch_files(job.work_dir, 0)

            timestamp = job.get_artifact_last_update()
            assert timestamp is not None
            assert isinstance(timestamp, float)
        finally:
            os.chdir(cwd)
