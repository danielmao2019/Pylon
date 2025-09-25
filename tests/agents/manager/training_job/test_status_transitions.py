from agents.manager.training_job import TrainingJob



def test_status_trainer_failed_no_logs_no_process(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('failed.py', {'epochs': 10})
    for i in range(2):
        make_trainer_epoch('failed', i)
    job = TrainingJob('./configs/failed.py')
    job.populate(epochs=10, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'failed'





def test_status_trainer_stuck_with_process_no_logs(temp_manager_root, write_config, make_trainer_epoch):
    from agents.monitor.process_info import ProcessInfo
    cfg = write_config('stuck.py', {'epochs': 10})
    for i in range(3):
        make_trainer_epoch('stuck', i)
    proc = ProcessInfo(pid='1', user='u', cmd='python main.py --config-filepath ./configs/stuck.py', start_time='t')
    job = TrainingJob('./configs/stuck.py')
    job.populate(epochs=10, config_to_process_info={'./configs/stuck.py': proc}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'stuck'





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





def test_status_trainer_finished_no_recent_logs(temp_manager_root, write_config, make_trainer_epoch):
    cfg = write_config('finished.py', {'epochs': 3})
    for i in range(3):
        make_trainer_epoch('finished', i)
    job = TrainingJob('./configs/finished.py')
    job.populate(epochs=3, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'finished'





def test_status_trainer_outdated_epochs(temp_manager_root, write_config, make_trainer_epoch):
    import time, os
    cfg = write_config('outdated.py', {'epochs': 1})
    make_trainer_epoch('outdated', 0)
    work = './logs/outdated/epoch_0'
    old = time.time() - (31 * 24 * 60 * 60)
    for name in ['validation_scores.json', 'optimizer_buffer.json', 'training_losses.pt']:
        p = os.path.join(work, name)
        os.utime(p, (old, old))
    job = TrainingJob('./configs/outdated.py')
    job.populate(epochs=1, config_to_process_info={}, sleep_time=1, outdated_days=30, force_progress_recompute=False)
    assert job.status == 'outdated'
