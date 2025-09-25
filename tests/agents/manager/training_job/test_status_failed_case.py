from agents.manager.training_job import TrainingJob



def test_status_trainer_failed_no_logs_no_process(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('failed.py', {'epochs': 10})
    for i in range(2):
        make_trainer_epoch('failed', i)
    job = TrainingJob('./configs/failed.py')
    job.populate(epochs=10, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'failed'
