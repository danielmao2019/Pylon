from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams


def test_status_trainer_outdated_epochs(temp_manager_root, write_config, make_trainer_epoch):
    import time, os
    cfg = write_config('outdated.py', {'epochs': 1})
    make_trainer_epoch('outdated', 0)
    work = './logs/outdated/epoch_0'
    old = time.time() - (31 * 24 * 60 * 60)
    for name in ['validation_scores.json', 'optimizer_buffer.json', 'training_losses.pt']:
        p = os.path.join(work, name)
        os.utime(p, (old, old))
    job = TrainingJob('python main.py --config-filepath ./configs/outdated.py')
    job.configure(
        JobRuntimeParams(
            epochs=1,
            sleep_time=1,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'outdated'
