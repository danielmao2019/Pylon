import pytest

from agents.manager.training_job import TrainingJob


def make_job(write_config, relative_path: str) -> TrainingJob:
    config_path = write_config(relative_path, {'epochs': 1})
    command = f'python main.py --config-filepath {config_path}'
    return TrainingJob(command)


def test_tmux_session_name_relative_to_benchmarks(temp_manager_root, write_config):
    job = make_job(write_config, 'benchmarks/foo/test_config.py')
    assert job.tmux_session_name() == 'foo/test_config.py'


def test_tmux_session_name_rejects_outside_benchmarks(temp_manager_root, write_config):
    job = make_job(write_config, 'custom/experiment.py')
    with pytest.raises(AssertionError):
        job.tmux_session_name()
