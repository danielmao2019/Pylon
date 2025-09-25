from agents.manager.training_job import TrainingJob

def test_status_trainer_running_with_recent_log(temp_manager_root, write_config, make_trainer_epoch, touch_log):
    cfg = write_config('running.py', {'epochs': 10})
    # Partial progress
    for i in range(3):
        make_trainer_epoch('running', i)
    # Fresh log within sleep_time window signals running
    touch_log('running', age_seconds=0)
    job = TrainingJob('./configs/running.py')
    job.populate(epochs=10, config_to_process_info={}, sleep_time=3600, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'running'
