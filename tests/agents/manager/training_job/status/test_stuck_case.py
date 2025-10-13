from agents.manager.training_job import TrainingJob
from agents.manager.runtime import JobRuntimeParams
from agents.monitor.process_info import ProcessInfo


def test_status_trainer_stuck_with_process_no_logs(
    temp_manager_root, write_config, make_trainer_epoch
):
    from agents.monitor.process_info import ProcessInfo

    cfg = write_config('stuck.py', {'epochs': 10})
    for i in range(3):
        make_trainer_epoch('stuck', i)
    command = 'python main.py --config-filepath ./configs/stuck.py'
    proc = ProcessInfo(pid='1', user='u', cmd=command, start_time='t')
    job = TrainingJob(command)
    job.configure(
        JobRuntimeParams(
            epochs=10,
            sleep_time=1,
            outdated_days=30,
            command_processes={command: proc},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'stuck'
