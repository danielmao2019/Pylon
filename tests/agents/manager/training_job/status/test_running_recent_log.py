from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams


def test_status_trainer_running_with_recent_log(
    temp_manager_root, write_config, make_trainer_epoch, touch_log
):
    cfg = write_config('running.py', {'epochs': 10})
    # Partial progress
    for i in range(3):
        make_trainer_epoch('running', i)
    # Fresh log within sleep_time window signals running
    touch_log('running', age_seconds=0)
    job = TrainingJob('python main.py --config-filepath ./configs/running.py')
    job.configure(
        JobRuntimeParams(
            epochs=10,
            sleep_time=3600,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'running'
