"""
Extended job status permutations using real classes and real dummy files.

Focus: additional stuck/failed variants beyond the core cases.
"""

import os
import time

from agents.manager.training_job import TrainingJob
from agents.manager.evaluation_job import EvaluationJob
from agents.manager.runtime import JobRuntimeParams
from agents.monitor.process_info import ProcessInfo


def test_trainer_stuck_with_stale_log_and_process(
    temp_manager_root, write_config, make_trainer_epoch, touch_log
):
    # Partially complete, stale log, process mapped -> stuck
    write_config('stuck_stale.py', {'epochs': 10})
    for i in range(2):
        make_trainer_epoch('stuck_stale', i)
    # Stale log (older than sleep_time window)
    log_path = touch_log('stuck_stale', age_seconds=7200)  # 2 hours old
    command = 'python main.py --config-filepath ./configs/stuck_stale.py'
    proc = ProcessInfo(pid='9', user='u', cmd=command, start_time='t')
    job = TrainingJob(command)
    job.configure(
        JobRuntimeParams(
            epochs=10,
            sleep_time=3600,
            outdated_days=30,
            command_processes={command: proc},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'stuck'


def test_trainer_failed_no_epochs_no_logs(temp_manager_root, write_config):
    # No epochs, no logs, no processes -> failed
    write_config('failed_empty.py', {'epochs': 5})
    job = TrainingJob('python main.py --config-filepath ./configs/failed_empty.py')
    job.configure(
        JobRuntimeParams(
            epochs=5,
            sleep_time=3600,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'failed'


def test_evaluator_failed_empty_eval_file(temp_manager_root, write_config):
    # evaluation_scores.json exists but is empty -> failed (treated as incomplete)
    write_config('eval_empty.py', {})
    work = os.path.join('./logs', 'eval_empty')
    os.makedirs(work, exist_ok=True)
    open(os.path.join(work, 'evaluation_scores.json'), 'w').close()  # zero-byte
    job = EvaluationJob('python main.py --config-filepath ./configs/eval_empty.py')
    job.configure(
        JobRuntimeParams(
            epochs=1,
            sleep_time=3600,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'failed'


def test_trainer_failed_non_matching_log_name(
    temp_manager_root, write_config, make_trainer_epoch
):
    # Log present but doesn't match pattern -> not considered running
    write_config('failed_logname.py', {'epochs': 10})
    for i in range(3):
        make_trainer_epoch('failed_logname', i)
    work = os.path.join('./logs', 'failed_logname')
    os.makedirs(work, exist_ok=True)
    # Create a log with non-matching name
    with open(os.path.join(work, 'other.log'), 'a'):
        os.utime(os.path.join(work, 'other.log'), None)
    job = TrainingJob('python main.py --config-filepath ./configs/failed_logname.py')
    job.configure(
        JobRuntimeParams(
            epochs=10,
            sleep_time=3600,
            outdated_days=30,
            command_processes={},
            force_progress_recompute=False,
        )
    )
    assert job.status == 'failed'
