# runners/trainers — tests structure

## Tests implementation structure

`tests/runners/trainers/test_interrupt_and_resume.py`

```text
test_interrupt_and_resume.py
├── import copy
├── import os
├── import threading
├── import time
├── from pathlib import Path
├── from typing import Dict
├── from agents.manager.training_job import TrainingJob
├── from configs.examples.linear.config import config as base_config
├── from runners.trainers.supervised_single_task_trainer import SupervisedSingleTaskTrainer
├── from utils.determinism import set_seed
├── def test_interrupt_and_resume() -> None
│   ├── # Interrupting a run mid-training and re-running resumes from the last finished-epoch checkpoint rather than recomputing it.
│   ├── calls _prepare_workspace
│   ├── calls _build_config
│   └── impls run the trainer twice, asserting the second run resumes from the saved checkpoint
├── def _prepare_workspace(base: Path) -> Dict[str, str]
│   ├── # Prepares the temp workspace directories for the interrupt-and-resume test.
│   ├── impls logs_dir = the logs directory under base
│   ├── impls configs_dir = the configs directory under base
│   ├── impls create logs_dir with its parents, tolerating one that already exists
│   ├── impls create configs_dir with its parents, tolerating one that already exists
│   ├── impls collect logs_dir and configs_dir as strings, keyed logs and configs  # impls-node-one-step:skip — one mapping; the list names its two entries
│   └── return  # that mapping
├── def _build_config(base: Path, work_dir: str) -> dict
│   ├── # Builds the SupervisedSingleTaskTrainer config for the interrupt-and-resume test.
│   ├── impls cfg = a deep copy of base_config
│   ├── impls cfg['work_dir'] = work_dir
│   ├── impls cfg['checkpoint_method'] = 'all'  # every epoch is checkpointed, so an interrupted run has one to resume from wherever it stopped
│   ├── impls cfg['log_dir'] = work_dir
│   ├── impls cfg['config_dir'] = the configs directory under base, as a string
│   └── return cfg
└── def train_until_epoch(config: dict, start_epoch: int, end_epoch: int) -> None
    ├── # Runs one trainer over epochs start_epoch up to end_epoch, beside a watcher that flags the epoch loop to stop once the range's last epoch is finished on disk.
    ├── calls SupervisedSingleTaskTrainer(config=config)  # -> trainer
    ├── calls trainer._init_components_
    ├── assert trainer.cum_epochs == start_epoch  # f"Expected to start from epoch {start_epoch}, but got {trainer.cum_epochs}"
    ├── impls print the epoch the run starts from
    ├── impls stop_observing = a threading.Event the watcher exits on
    ├── impls stop_training = a threading.Event the epoch loop breaks on
    ├── def observer_thread() [local]
    │   ├── # Watches for the range's last epoch to finish on disk, then sets stop_training for the epoch loop to read at its next check.
    │   └── while stop_observing is not set
    │       ├── impls epoch_dir = the epoch_{end_epoch - 1} directory under config['work_dir']
    │       ├── if epoch_dir exists and TrainingJob._check_epoch_finished(epoch_dir, trainer.expected_files, check_load=True) reports it finished
    │       │   ├── impls set stop_training
    │       │   └── break
    │       └── impls sleep a tenth of a second
    ├── impls observer = a threading.Thread running observer_thread
    ├── impls start observer
    ├── calls trainer.logger.page_break
    ├── for each idx in range(start_epoch, end_epoch)
    │   ├── if stop_training is set
    │   │   └── break
    │   ├── calls set_seed(seed=trainer.train_seeds[idx])
    │   ├── calls trainer._train_epoch_
    │   ├── calls trainer._val_epoch_
    │   ├── calls trainer.logger.page_break
    │   ├── impls trainer.cum_epochs = idx + 1
    │   └── impls sleep a second  # leaves the watcher a window to set stop_training before the next epoch starts
    ├── if trainer.after_train_thread exists and is alive
    │   └── impls join it
    ├── if trainer.after_val_thread exists and is alive
    │   └── impls join it
    ├── impls set stop_observing
    ├── impls join observer
    └── impls delete trainer
```
