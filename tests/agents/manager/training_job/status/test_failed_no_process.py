from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams

def test_status_trainer_failed_no_logs_no_process(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('failed.py', {'epochs': 10})
    for i in range(2):
        make_trainer_epoch('failed', i)
    job = TrainingJob('python main.py --config-filepath ./configs/failed.py')
    job.configure(
        JobRuntimeParams(
            epochs=10,
            sleep_time=1,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'failed'
