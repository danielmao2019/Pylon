# runners/trainers — code implementation structure

## Code implementation structure trees

`runners/trainers/base_trainer.py`

```text
base_trainer.py
├── from typing import Any, Dict, List, Optional
├── from abc import ABC, abstractmethod
├── import copy
├── import os
├── import glob
├── import time
├── import json
├── import jsbeautifier
├── import torch
├── import threading
├── import criteria
├── from concurrent.futures import ThreadPoolExecutor, as_completed
├── from utils.builders import build_from_config
├── from utils.determinism import set_determinism, set_seed
├── from utils.io.json import serialize_tensor, save_json
├── from agents.manager.training_job import TrainingJob
├── from agents.monitor.system_monitor import SystemMonitor
├── from utils.dynamic_executor import create_dynamic_executor
├── from utils.logging.text_logger import TextLogger
├── from utils.logging.screen_logger import ScreenLogger
├── from utils.logging import echo_page_break, log_losses, log_scores
├── from runners.model_comparison import compare_scores, get_metric_directions
└── class BaseTrainer(ABC)
    ├── # Abstract trainer: builds every component from one config dict and drives the train/val/test epoch loop with checkpoint resume.
    ├── def __init__(self, config: dict, device: Optional[torch.device] = torch.device('cuda')) -> None
    │   ├── # Deep-copies the config, stores the device, and initializes the work dir, epoch count, and threading primitives.
    │   ├── impls copy.deepcopy the config onto self.config
    │   ├── impls store device / eval_n_jobs
    │   ├── calls self._init_work_dir
    │   ├── calls self._init_tot_epochs
    │   ├── impls enable torch.autograd anomaly detection
    │   └── impls init the after-train / after-val threads and the threading.Lock buffer lock  # impls-node-one-step:skip
    ├── def run(self) -> None
    │   ├── # Initializes components, runs the train/val epoch loop with early stopping, then the test epoch.
    │   ├── calls self._init_components_
    │   ├── for idx in range(start_epoch, self.tot_epochs)
    │   │   ├── if self.early_stopping and self.early_stopping.should_stop()
    │   │   │   ├── calls self._save_progress
    │   │   │   └── break
    │   │   ├── calls set_seed
    │   │   ├── calls self._train_epoch_
    │   │   ├── calls self._val_epoch_
    │   │   └── calls self._save_progress
    │   ├── impls join any remaining after-train / after-val threads
    │   └── calls self._test_epoch_
    ├── def _init_components_(self) -> None
    │   ├── # Initializes every trainer component in dependency order.
    │   ├── calls self._init_logger
    │   ├── calls self._init_determinism
    │   ├── calls self._init_checkpoint_indices
    │   ├── calls self._init_state
    │   ├── calls self._init_dataloaders
    │   ├── calls self._init_criterion
    │   ├── calls self._init_metric
    │   ├── calls self._init_model
    │   ├── calls self._init_optimizer
    │   ├── calls self._init_scheduler
    │   ├── calls self._init_debugger
    │   ├── calls self._init_early_stopping
    │   └── calls self._load_checkpoint
    ├── def _train_epoch_(self) -> None
    │   ├── # Runs one training epoch, skipping when no dataloader/model or the epoch's checkpoint already exists.
    │   ├── if not (self.train_dataloader and self.model)
    │   │   └── return
    │   ├── if the epoch's checkpoint already exists
    │   │   ├── impls torch.load the found epoch checkpoint
    │   │   ├── impls load_state_dict the model / optimizer / scheduler from it
    │   │   └── return
    │   ├── impls join the previous after-train thread if alive
    │   ├── calls self._before_train_loop
    │   ├── for each dp in self.train_dataloader
    │   │   └── calls self._train_step
    │   └── calls self._after_train_loop_
    ├── def _val_epoch_(self) -> None
    │   ├── # Runs one validation epoch, sequentially or via a dynamic parallel executor, then the after-val hook.
    │   ├── if not (self.val_dataloader and self.model)
    │   │   └── return
    │   ├── impls join the previous after-val thread if alive
    │   ├── calls self._before_val_loop
    │   ├── if self.eval_n_jobs == 1
    │   │   └── for each dp in self.val_dataloader
    │   │       └── calls self._eval_step
    │   ├── else
    │   │   ├── calls create_dynamic_executor
    │   │   ├── impls submit each self._eval_step to the executor
    │   │   └── impls collect the submitted futures via as_completed
    │   └── calls self._after_val_loop_
    ├── @torch.no_grad() def _test_epoch_(self) -> None
    │   ├── # Runs the test epoch on the best checkpoint: before-test setup, the test loop, and after-test save.
    │   ├── if not (self.test_dataloader and self.model)
    │   │   └── return
    │   ├── calls self._before_test_loop_
    │   ├── for each dp in self.test_dataloader
    │   │   └── calls self._eval_step
    │   └── calls self._after_test_loop_
    ├── def _init_work_dir(self) -> None
    │   ├── # Creates and stores the work dir from config, or None when unconfigured.
    │   ├── if self.config.get('work_dir', None)
    │   │   ├── impls os.makedirs the work_dir
    │   │   └── impls store the work_dir
    │   └── else
    │       └── impls set self.work_dir to None
    ├── def _init_tot_epochs(self) -> None
    │   ├── # Reads and stores the total epoch count from config.
    │   └── impls store self.tot_epochs from config['epochs']
    ├── def _save_progress(self) -> None
    │   ├── # Writes completed-epoch count, percentage, and early-stop status to progress.json.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── if self.early_stopping and self.early_stopping.should_stop_early
    │   │   └── impls mark early_stopped at self.cum_epochs
    │   └── calls save_json  # the progress dict to progress.json
    ├── def _init_logger(self) -> None
    │   ├── # Initializes the git log, training log (screen logger with text-logger fallback), config dump, and system monitor.
    │   ├── impls count prior train_val logs via glob.glob for the session index
    │   ├── impls os.system the git branch / status / log into the git log
    │   ├── calls echo_page_break  # the git-log section headings
    │   ├── try
    │   │   └── calls ScreenLogger
    │   ├── except Exception
    │   │   └── calls TextLogger
    │   ├── impls jsbeautifier-dump the config to config.json
    │   ├── calls SystemMonitor
    │   └── impls start the system monitor
    ├── def _init_determinism(self) -> None
    │   ├── # Seeds determinism and validates/stores the per-epoch train/val seeds, test seed, and init seed.
    │   ├── calls set_determinism
    │   ├── impls validate the train_seeds / val_seeds / test_seed from config
    │   ├── impls store the train_seeds / val_seeds / test_seed
    │   └── calls set_seed  # the init seed
    ├── def _init_checkpoint_indices(self) -> None
    │   ├── # Precomputes the epoch indices at which checkpoints and debug outputs are saved, by checkpoint_method.
    │   ├── if checkpoint_method == 'all'
    │   │   └── impls checkpoint_indices = every epoch index
    │   ├── elif checkpoint_method == 'latest'
    │   │   └── impls checkpoint_indices = the last epoch only
    │   └── else
    │       └── impls checkpoint_indices = every-N epochs plus the last
    ├── def _init_state(self) -> None
    │   ├── # Determines the resume point self.cum_epochs by scanning finished epoch dirs that carry a checkpoint.
    │   ├── if self.work_dir is None
    │   │   ├── impls set cum_epochs to 0
    │   │   └── return
    │   ├── for idx in range(self.tot_epochs)
    │   │   ├── calls TrainingJob._check_epoch_finished
    │   │   ├── if not epoch_finished
    │   │   │   └── break
    │   │   └── impls record load_idx when the epoch dir has a checkpoint
    │   ├── if load_idx is None
    │   │   ├── impls set cum_epochs to 0
    │   │   └── return
    │   └── impls set cum_epochs to load_idx + 1
    ├── def _init_dataloaders(self) -> None
    │   ├── # Builds the train/val/test dataloaders from config (val/test batch size defaulting to 1), or None when unconfigured.
    │   ├── if self.config.get('train_dataset', None) and self.config.get('train_dataloader', None)
    │   │   ├── calls build_from_config  # the train dataset
    │   │   └── calls build_from_config  # the train dataloader, shuffle=True
    │   ├── else
    │   │   └── impls set train_dataloader to None
    │   ├── if self.config.get('val_dataset', None) and self.config.get('val_dataloader', None)
    │   │   ├── calls build_from_config  # the val dataset
    │   │   └── calls build_from_config  # the val dataloader, shuffle=False
    │   ├── else
    │   │   └── impls set val_dataloader to None
    │   ├── if self.config.get('test_dataset', None) and self.config.get('test_dataloader', None)
    │   │   ├── calls build_from_config  # the test dataset
    │   │   └── calls build_from_config  # the test dataloader, shuffle=False
    │   └── else
    │       └── impls set test_dataloader to None
    ├── def _init_criterion(self) -> None
    │   ├── # Builds the criterion from config, asserts it is a criteria.BaseCriterion nn.Module, moves it to device, or None.
    │   ├── if self.config.get('criterion', None)
    │   │   ├── calls build_from_config
    │   │   ├── impls assert isinstance criteria.BaseCriterion and torch.nn.Module  # impls-node-one-step:skip
    │   │   └── impls move the criterion to self.device
    │   └── else
    │       └── impls set self.criterion to None
    ├── def _init_metric(self) -> None
    │   ├── # Builds the metric from config, or None when unconfigured.
    │   ├── if self.config.get('metric', None)
    │   │   └── calls build_from_config
    │   └── else
    │       └── impls set self.metric to None
    ├── def _init_model(self) -> None
    │   ├── # Builds the model from config, asserts it is an nn.Module, moves it to device, or None.
    │   ├── if self.config.get('model', None)
    │   │   ├── calls build_from_config
    │   │   ├── impls assert isinstance torch.nn.Module
    │   │   └── impls move the model to self.device
    │   └── else
    │       └── impls set self.model to None
    ├── @abstractmethod def _init_optimizer(self) -> None
    │   ├── # Abstract hook: subclasses build self.optimizer from config.
    │   └── raise NotImplementedError
    ├── @abstractmethod def _init_scheduler(self) -> None
    │   ├── # Abstract hook: subclasses build self.scheduler from config.
    │   └── raise NotImplementedError
    ├── def _init_debugger(self)
    │   ├── # Builds the debugger from config and registers its forward hooks, or None.
    │   ├── if self.config.get('debugger', None)
    │   │   └── calls build_from_config  # the debugger, model=self.model
    │   └── else
    │       └── impls set self.debugger to None
    ├── def _init_early_stopping(self) -> None
    │   ├── # Builds the early-stopping object from config (or None) and updates it with existing scores.
    │   ├── if early_stopping_config is None
    │   │   ├── impls set self.early_stopping to None
    │   │   └── return
    │   ├── calls build_from_config  # the early stopping, with work_dir / tot_epochs / metric / expected_files / logger
    │   └── impls update early_stopping with existing scores
    ├── def _load_checkpoint(self) -> None
    │   ├── # Loads the model/optimizer/scheduler state_dicts from the last finished epoch's checkpoint when resuming.
    │   ├── if self.cum_epochs == 0
    │   │   └── return
    │   ├── impls torch.load the previous epoch's checkpoint
    │   └── impls load_state_dict into self.model, self.optimizer, and self.scheduler  # impls-node-one-step:skip
    ├── def _before_train_loop(self) -> None
    │   ├── # Puts the model in train mode, resets criterion/optimizer/logger buffers, and sets the epoch's train seed.
    │   ├── impls model.train
    │   ├── impls reset the criterion / optimizer / logger buffers
    │   └── impls set the train dataloader's base seed to this epoch's train seed
    ├── def _train_step(self, dp: Dict[str, Dict[str, Any]]) -> None
    │   ├── # Runs one training iteration: forward, loss, logging, gradient set, optimizer + scheduler step.
    │   ├── calls self.model      # dp['inputs'] → dp['outputs']
    │   ├── calls self.criterion  # y_pred=outputs, y_true=labels → dp['losses']
    │   ├── calls log_losses
    │   ├── calls self._set_gradients_
    │   ├── impls optimizer.step
    │   ├── impls scheduler.step
    │   └── impls log the iteration time via time.time
    ├── @abstractmethod def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None
    │   ├── # Abstract hook: subclasses zero grads and backprop the datapoint's losses.
    │   └── raise NotImplementedError
    ├── def _after_train_loop_(self) -> None
    │   ├── # Spawns a background thread that saves the epoch's losses, optimizer buffer, and checkpoint, and relinks checkpoint_latest.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── def after_train_ops() [local]
    │   │   ├── # Saves the epoch root's training_losses, optimizer_buffer, and checkpoint, then relinks checkpoint_latest.
    │   │   ├── impls os.makedirs the epoch root
    │   │   ├── impls summarize the criterion / optimizer buffers to disk
    │   │   ├── calls self._save_checkpoint_
    │   │   └── impls relink checkpoint_latest.pt to the epoch checkpoint
    │   ├── calls after_train_ops  # the threading.Thread target
    │   └── impls start the after-train thread
    ├── def _save_checkpoint_(self, output_path: str) -> None
    │   ├── # Default checkpoint save: torch.save the model/optimizer/scheduler state_dicts to output_path.
    │   └── impls torch.save the model / optimizer / scheduler state_dicts to output_path
    ├── def _before_val_loop(self) -> None
    │   ├── # Puts the model in eval mode, resets metric/logger, sets the val seed, and toggles the debugger by checkpoint index.
    │   ├── impls model.eval
    │   ├── impls reset the metric / logger
    │   ├── impls set the val base seed
    │   ├── if self.debugger and self.cum_epochs in self.checkpoint_indices
    │   │   ├── impls enable the debugger
    │   │   └── impls reset the debugger
    │   └── elif self.debugger
    │       └── impls disable the debugger
    ├── def _eval_step(self, dp: Dict[str, Dict[str, Any]], flush_prefix: Optional[str] = None) -> None
    │   ├── # Runs one eval iteration: inference, metric, optional debug outputs, score logging, optional flush.
    │   ├── calls self.model   # dp['inputs'] → dp['outputs']
    │   ├── calls self.metric  # dp → dp['scores']
    │   ├── if self.debugger and self.debugger.enabled
    │   │   └── calls self.debugger  # dp, self.model → dp['debug']
    │   ├── calls log_scores
    │   └── if flush_prefix is not None
    │       └── impls flush the logger with flush_prefix
    ├── def _after_val_loop_(self) -> None
    │   ├── # Spawns a background thread that saves validation scores, updates early stopping, relinks the best checkpoint, saves debug outputs, and cleans checkpoints.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── def after_val_ops() [local]
    │   │   ├── # Saves the epoch's validation scores and best-checkpoint link and cleans old checkpoints.
    │   │   ├── impls os.makedirs the epoch root
    │   │   ├── impls summarize the metric scores to disk
    │   │   ├── impls update early stopping when configured
    │   │   ├── try
    │   │   │   └── calls self._find_best_checkpoint
    │   │   ├── except
    │   │   │   └── impls set best_checkpoint to None
    │   │   ├── impls save debugger outputs when enabled
    │   │   └── calls self._clean_checkpoints
    │   ├── calls after_val_ops  # the threading.Thread target
    │   └── impls start the after-val thread
    ├── def _before_test_loop_(self) -> str
    │   ├── # Loads the best checkpoint into the model, sets eval mode and the test seed, and returns the checkpoint path.
    │   ├── calls self._find_best_checkpoint
    │   ├── impls torch.load the best checkpoint
    │   ├── impls load_state_dict the best checkpoint
    │   ├── impls set the model to eval mode
    │   ├── impls set the test base seed
    │   └── return  # the best checkpoint path
    ├── def _after_test_loop_(self, best_checkpoint: str) -> None
    │   ├── # Writes the test scores and best-checkpoint path to test/test_results.json.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── calls serialize_tensor  # the metric summary
    │   └── impls jsbeautifier-dump the results to test_results.json
    ├── def _find_best_checkpoint(self) -> str
    │   ├── # Scans finished epochs and returns the checkpoint path with the best validation score.
    │   ├── calls get_metric_directions
    │   ├── while epoch_idx < self.tot_epochs
    │   │   ├── calls TrainingJob._check_epoch_finished
    │   │   ├── if not finished
    │   │   │   └── break
    │   │   ├── impls json.load the epoch's validation_scores
    │   │   ├── if best_scores is None
    │   │   │   └── impls set this epoch as best
    │   │   └── else
    │   │       ├── calls compare_scores
    │   │       └── impls update best when current is better
    │   ├── if best_epoch_dir is None
    │   │   └── raise ValueError
    │   └── return  # the best epoch's checkpoint path
    ├── def _clean_checkpoints(self, latest_checkpoint: str, best_checkpoint: Optional[str] = None) -> None
    │   ├── # Removes epoch checkpoints outside the keep-set (latest, best, precomputed indices), in parallel.
    │   ├── if checkpoint_method == 'all'
    │   │   └── return
    │   ├── impls build the keep-set from latest, best, and checkpoint_indices  # impls-node-one-step:skip
    │   ├── impls glob.glob the existing epoch checkpoints
    │   ├── def clean_single_checkpoint(checkpoint: str) -> None [local]
    │   │   ├── # Removes one checkpoint when it is outside the keep-set and its next epoch has finished.
    │   │   ├── if checkpoint in keep_checkpoints
    │   │   │   └── return
    │   │   ├── calls TrainingJob._check_epoch_finished
    │   │   └── impls os.system rm the checkpoint when the next epoch finished
    │   ├── calls clean_single_checkpoint  # mapped over the existing checkpoints
    │   └── impls run clean_single_checkpoint across a ThreadPoolExecutor
    └── @property def expected_files(self) -> List[str]
        ├── # The per-epoch artifact filenames that mark an epoch finished.
        └── return  # ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
```

`runners/trainers/supervised_single_task_trainer.py`

```text
supervised_single_task_trainer.py
├── from typing import Any, Dict
├── import torch
├── from optimizers.single_task_optimizer import SingleTaskOptimizer
├── from runners.trainers.base_trainer import BaseTrainer
├── from utils.builders import build_from_config, build_scheduler
└── class SupervisedSingleTaskTrainer(BaseTrainer)
    ├── # Supervised single-task trainer: one optimizer over the model params, single scalar-loss backprop, no-grad validation.
    ├── def _init_optimizer(self) -> None   [override]
    │   ├── # Builds the single-task optimizer over the model parameters, or skips when unconfigured.
    │   ├── if not self.config.get('optimizer', None)
    │   │   └── return
    │   ├── impls set optimizer_config's params to list(self.model.parameters())
    │   └── calls build_from_config  # the optimizer
    ├── def _init_scheduler(self)   [override]
    │   ├── # Builds the LR scheduler over the single-task optimizer, or skips when unconfigured.
    │   ├── if not self.config.get('scheduler', None)
    │   │   └── return
    │   ├── impls assert the optimizer is a SingleTaskOptimizer wrapping a torch.optim.Optimizer
    │   └── calls build_scheduler
    ├── def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None   [override]
    │   ├── # Zeros grads and backprops the single scalar loss.
    │   ├── impls optimizer.zero_grad
    │   ├── impls assert dp['losses'] is a single-element torch.Tensor
    │   └── impls losses.backward
    └── @torch.no_grad() def _val_epoch_(self) -> None   [override]
        ├── # Runs the base validation epoch under torch.no_grad to prevent gradient computation.
        └── calls super()._val_epoch_
```
