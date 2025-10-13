from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams


def test_status_trainer_finished_no_recent_logs(
    temp_manager_root, write_config, make_trainer_epoch
):
    cfg = write_config('finished.py', {'epochs': 3})
    for i in range(3):
        make_trainer_epoch('finished', i)
    job = TrainingJob('python main.py --config-filepath ./configs/finished.py')
    job.configure(
        JobRuntimeParams(
            epochs=3,
            sleep_time=1,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'finished'
