# runners/trainers — code implementation structure

## Code implementation structure trees

`runners/trainers/base_trainer.py`

```text
base_trainer.py
├── from typing import List, Dict, Any, Optional
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
├── from utils.io.json import serialize_tensor
├── from utils.io.json import save_json
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
    │   ├── impls self.device = device
    │   ├── impls self.eval_n_jobs = self.config.get('eval_n_jobs', 1)
    │   ├── calls self._init_work_dir
    │   ├── calls self._init_tot_epochs
    │   ├── impls enable torch.autograd anomaly detection
    │   ├── impls set the after-train / after-val threads to None
    │   └── impls self.buffer_lock = threading.Lock()
    ├── def run(self) -> None
    │   ├── # Initializes components, runs the train/val epoch loop with early stopping, then the test epoch.
    │   ├── calls self._init_components_
    │   ├── impls start_epoch = self.cum_epochs
    │   ├── for idx in range(start_epoch, self.tot_epochs)
    │   │   ├── if self.early_stopping and self.early_stopping.should_stop()
    │   │   │   ├── calls self._save_progress
    │   │   │   └── break
    │   │   ├── calls set_seed  # seed=self.train_seeds[idx]
    │   │   ├── calls self._train_epoch_
    │   │   ├── calls self._val_epoch_
    │   │   ├── impls self.cum_epochs = idx + 1
    │   │   └── calls self._save_progress
    │   ├── if self.after_train_thread and self.after_train_thread.is_alive()
    │   │   └── impls join the after-train thread
    │   ├── if self.after_val_thread and self.after_val_thread.is_alive()
    │   │   └── impls join the after-val thread
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
    │   │   ├── impls checkpoint = torch.load of the epoch's checkpoint
    │   │   ├── impls load_state_dict the model / optimizer / scheduler from it
    │   │   └── return
    │   ├── if self.after_train_thread and self.after_train_thread.is_alive()
    │   │   └── impls join the after-train thread
    │   ├── calls self._before_train_loop
    │   ├── for each dp in self.train_dataloader
    │   │   └── calls self._train_step
    │   └── calls self._after_train_loop_
    ├── def _val_epoch_(self) -> None
    │   ├── # Runs one validation epoch, sequentially or via a dynamic parallel executor, then the after-val hook.
    │   ├── if not (self.val_dataloader and self.model)
    │   │   └── return
    │   ├── if self.after_val_thread and self.after_val_thread.is_alive()
    │   │   └── impls join the after-val thread
    │   ├── calls self._before_val_loop
    │   ├── if self.eval_n_jobs == 1
    │   │   └── for each dp in self.val_dataloader
    │   │       └── calls self._eval_step
    │   ├── else
    │   │   ├── impls max_workers = self.eval_n_jobs if self.eval_n_jobs > 1 else None
    │   │   ├── calls create_dynamic_executor  # max_workers, min_workers=1 → executor
    │   │   └── with executor
    │   │       ├── impls future_to_args = each submitted self._eval_step future
    │   │       └── for future in as_completed(future_to_args)
    │   │           └── impls future.result, re-raising any worker exception
    │   └── calls self._after_val_loop_
    ├── @torch.no_grad() def _test_epoch_(self) -> None
    │   ├── # Runs the test epoch on the best checkpoint: before-test setup, the test loop, and after-test save.
    │   ├── if not (self.test_dataloader and self.model)
    │   │   └── return
    │   ├── calls self._before_test_loop_  # → best_checkpoint
    │   ├── for each dp in self.test_dataloader
    │   │   └── calls self._eval_step
    │   └── calls self._after_test_loop_  # best_checkpoint=best_checkpoint
    ├── def _init_work_dir(self) -> None
    │   ├── # Creates and stores the work dir from config, or None when unconfigured.
    │   ├── if self.config.get('work_dir', None)
    │   │   ├── impls work_dir = config['work_dir']
    │   │   ├── impls os.makedirs the work_dir
    │   │   └── impls store it on self.work_dir
    │   └── else
    │       └── impls set self.work_dir to None
    ├── def _init_tot_epochs(self) -> None
    │   ├── # Reads and stores the total epoch count from config.
    │   ├── impls tot_epochs = config['epochs']
    │   └── impls store it on self.tot_epochs
    ├── def _save_progress(self) -> None
    │   ├── # Writes completed-epoch count, percentage, and early-stop status to progress.json.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── impls early_stopped = False
    │   ├── impls early_stopped_at_epoch = None
    │   ├── if self.early_stopping and self.early_stopping.should_stop_early
    │   │   └── impls mark early_stopped at self.cum_epochs
    │   ├── impls progress_file = work_dir's progress.json
    │   ├── impls progress_data = the completed-epoch count, percentage, and early-stop status  # impls-node-one-step:skip
    │   └── calls save_json  # the progress dict to progress.json
    ├── def _init_logger(self) -> None
    │   ├── # Initializes the git log, training log (screen logger with text-logger fallback), config dump, and system monitor.
    │   ├── impls session_idx = the count of prior train_val logs via glob.glob
    │   ├── impls git_log = the session's git log path under work_dir
    │   ├── calls echo_page_break  # each git-log section heading, written before its own command
    │   ├── impls os.system the git branch / status / log, each appending under the heading just written
    │   ├── impls log_filepath = the session's train_val log path under work_dir
    │   ├── try
    │   │   └── calls ScreenLogger  # max_iterations=10, filepath=log_filepath, layout="train" → self.logger
    │   ├── except Exception
    │   │   ├── impls print the fall-back message with the exception
    │   │   └── calls TextLogger  # filepath=log_filepath → self.logger
    │   ├── impls jsbeautifier-dump the config to config.json
    │   ├── calls SystemMonitor  # → self.system_monitor
    │   └── impls start the system monitor
    ├── def _init_determinism(self) -> None
    │   ├── # Seeds determinism, validates and stores the per-epoch train/val seeds and the test seed, and seeds once from the init seed.
    │   ├── calls set_determinism
    │   ├── impls train_seeds / val_seeds / test_seed = the config's seed entries  # impls-node-one-step:skip
    │   ├── impls validate the train_seeds / val_seeds / test_seed from config
    │   ├── impls store the train_seeds on self.train_seeds
    │   ├── impls store the val_seeds on self.val_seeds
    │   ├── impls store the test_seed on self.test_seed
    │   ├── impls init_seed = config['init_seed']
    │   └── calls set_seed  # the init seed
    ├── def _init_checkpoint_indices(self) -> None
    │   ├── # Precomputes the epoch indices at which checkpoints and debug outputs are saved, by checkpoint_method.
    │   ├── impls checkpoint_method = self.config.get('checkpoint_method', 'latest')
    │   ├── if checkpoint_method == 'all'
    │   │   └── impls self.checkpoint_indices = every epoch index
    │   ├── elif checkpoint_method == 'latest'
    │   │   └── impls self.checkpoint_indices = the last epoch only
    │   └── else
    │       ├── impls self.checkpoint_indices = every-N epochs
    │       └── if self.tot_epochs - 1 not in self.checkpoint_indices
    │           └── impls append the last epoch index
    ├── def _init_state(self) -> None
    │   ├── # Determines the resume point self.cum_epochs by scanning finished epoch dirs that carry a checkpoint, requiring the checkpoint itself on interval-checkpoint epochs.
    │   ├── if self.work_dir is None
    │   │   ├── impls set cum_epochs to 0
    │   │   └── return
    │   ├── impls load_idx = None
    │   ├── for idx in range(self.tot_epochs)
    │   │   ├── impls epoch_dir = work_dir's epoch_{idx}
    │   │   ├── calls TrainingJob._check_epoch_finished  # → epoch_finished
    │   │   ├── impls checkpoint_method = self.config.get('checkpoint_method', 'latest')
    │   │   ├── if isinstance(checkpoint_method, int) and checkpoint_method > 0
    │   │   │   └── if idx in self.checkpoint_indices
    │   │   │       └── impls narrow epoch_finished by whether the epoch dir holds checkpoint.pt
    │   │   ├── if not epoch_finished
    │   │   │   └── break
    │   │   └── if os.path.isfile the epoch dir's checkpoint.pt
    │   │       └── impls load_idx = idx
    │   ├── if load_idx is None
    │   │   ├── impls set cum_epochs to 0
    │   │   └── return
    │   └── impls set cum_epochs to load_idx + 1
    ├── def _init_dataloaders(self) -> None
    │   ├── # Builds the train/val/test dataloaders from config, or None when unconfigured, writing a default val/test batch size back into self.config.
    │   ├── if self.config.get('train_dataset', None) and self.config.get('train_dataloader', None)
    │   │   ├── calls build_from_config  # the train dataset → train_dataset
    │   │   └── calls build_from_config  # the train dataloader, dataset=train_dataset, shuffle=True → self.train_dataloader
    │   ├── else
    │   │   └── impls set self.train_dataloader to None
    │   ├── if self.config.get('val_dataset', None) and self.config.get('val_dataloader', None)
    │   │   ├── calls build_from_config  # the val dataset → val_dataset
    │   │   ├── if 'batch_size' not in self.config['val_dataloader']['args']
    │   │   │   └── impls set that batch_size to 1 in self.config
    │   │   └── calls build_from_config  # the val dataloader, dataset=val_dataset, shuffle=False → self.val_dataloader
    │   ├── else
    │   │   └── impls set self.val_dataloader to None
    │   ├── if self.config.get('test_dataset', None) and self.config.get('test_dataloader', None)
    │   │   ├── calls build_from_config  # the test dataset → test_dataset
    │   │   ├── if 'batch_size' not in self.config['test_dataloader']['args']
    │   │   │   └── impls set that batch_size to 1 in self.config
    │   │   └── calls build_from_config  # the test dataloader, dataset=test_dataset, shuffle=False → self.test_dataloader
    │   └── else
    │       └── impls set self.test_dataloader to None
    ├── def _init_criterion(self) -> None
    │   ├── # Builds the criterion from config, asserts it is a criteria.BaseCriterion nn.Module, moves it to device, or None.
    │   ├── if self.config.get('criterion', None)
    │   │   ├── calls build_from_config  # → criterion
    │   │   ├── impls assert isinstance criteria.BaseCriterion and torch.nn.Module  # impls-node-one-step:skip
    │   │   ├── impls move the criterion to self.device
    │   │   └── impls store it on self.criterion
    │   └── else
    │       └── impls set self.criterion to None
    ├── def _init_metric(self) -> None
    │   ├── # Builds the metric from config, or None when unconfigured.
    │   ├── if self.config.get('metric', None)
    │   │   └── calls build_from_config  # → self.metric
    │   └── else
    │       └── impls set self.metric to None
    ├── def _init_model(self) -> None
    │   ├── # Builds the model from config, asserts it is an nn.Module, moves it to device, or None.
    │   ├── if self.config.get('model', None)
    │   │   ├── calls build_from_config  # → model
    │   │   ├── impls assert isinstance torch.nn.Module
    │   │   ├── impls move the model to self.device
    │   │   └── impls store it on self.model
    │   └── else
    │       └── impls set self.model to None
    ├── @abstractmethod def _init_optimizer(self) -> None
    │   ├── # Abstract hook: subclasses build self.optimizer from config.
    │   └── raise NotImplementedError
    ├── @abstractmethod def _init_scheduler(self) -> None
    │   ├── # Abstract hook: subclasses build self.scheduler from config.
    │   └── raise NotImplementedError
    ├── def _init_debugger(self)
    │   ├── # Builds the debugger from config over self.model, or None when unconfigured.
    │   ├── if self.config.get('debugger', None)
    │   │   └── calls build_from_config  # the debugger, model=self.model → self.debugger
    │   └── else
    │       └── impls set self.debugger to None
    ├── def _init_early_stopping(self) -> None
    │   ├── # Builds the early-stopping object from config (or None) and updates it with existing scores.
    │   ├── impls early_stopping_config = self.config.get('early_stopping', None)
    │   ├── if early_stopping_config is None
    │   │   ├── impls set self.early_stopping to None
    │   │   └── return
    │   ├── calls build_from_config  # config=early_stopping_config, with work_dir / tot_epochs / metric / expected_files / logger → self.early_stopping
    │   └── impls update early_stopping with existing scores
    ├── def _load_checkpoint(self) -> None
    │   ├── # Loads the model/optimizer/scheduler state_dicts from the last finished epoch's checkpoint when resuming.
    │   ├── if self.cum_epochs == 0
    │   │   └── return
    │   ├── impls checkpoint_filepath = the previous epoch dir's checkpoint.pt
    │   ├── impls checkpoint = torch.load of the previous epoch's checkpoint
    │   └── impls load_state_dict into self.model, self.optimizer, and self.scheduler  # impls-node-one-step:skip
    ├── def _before_train_loop(self) -> None
    │   ├── # Puts the model and the logger in train mode, resets the criterion and optimizer buffers, and sets the epoch's train seed.
    │   ├── impls model.train
    │   ├── impls reset the criterion buffer
    │   ├── impls reset the optimizer buffer
    │   ├── impls put the logger in train mode
    │   └── impls set the train dataloader's base seed to this epoch's train seed
    ├── def _train_step(self, dp: Dict[str, Dict[str, Any]]) -> None
    │   ├── # Runs one training iteration: forward, loss, logging, gradient set, optimizer + scheduler step.
    │   ├── calls self.model      # dp['inputs'] → dp['outputs']
    │   ├── calls self.criterion  # y_pred=outputs, y_true=labels → dp['losses']
    │   ├── impls log the learning rate from self.scheduler.get_last_lr
    │   ├── calls log_losses
    │   ├── calls self._set_gradients_
    │   ├── impls optimizer.step
    │   ├── impls scheduler.step
    │   ├── calls self.system_monitor.log_stats  # the CPU / GPU stats onto the logger
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
    │   │   ├── impls epoch_root = work_dir's epoch_{self.cum_epochs}
    │   │   ├── impls os.makedirs the epoch root
    │   │   ├── with self.buffer_lock
    │   │   │   └── impls summarize the criterion buffer to training_losses.pt
    │   │   ├── impls torch.load training_losses.pt back to verify the write
    │   │   ├── impls summarize the optimizer buffer to optimizer_buffer.json
    │   │   ├── impls json.load optimizer_buffer.json back to verify the write
    │   │   ├── impls latest_checkpoint = the epoch root's checkpoint.pt
    │   │   ├── calls self._save_checkpoint_  # output_path=latest_checkpoint
    │   │   ├── impls soft_link = work_dir's checkpoint_latest.pt
    │   │   ├── if os.path.islink(soft_link)
    │   │   │   └── impls os.system rm the existing checkpoint_latest.pt link
    │   │   └── impls os.system ln -s the epoch checkpoint to checkpoint_latest.pt
    │   ├── calls after_train_ops  # the threading.Thread target
    │   ├── impls self.after_train_thread = threading.Thread(target=after_train_ops)
    │   └── impls start the after-train thread
    ├── def _save_checkpoint_(self, output_path: str) -> None
    │   ├── # Default checkpoint save: torch.save the model/optimizer/scheduler state_dicts to output_path.
    │   └── impls torch.save the model / optimizer / scheduler state_dicts to output_path
    ├── def _before_val_loop(self) -> None
    │   ├── # Puts the model and the logger in eval mode, resets the metric buffer, sets the val seed, and toggles the debugger by checkpoint index.
    │   ├── impls model.eval
    │   ├── impls reset the metric buffer
    │   ├── impls put the logger in eval mode
    │   ├── impls set the val dataloader's base seed to this epoch's val seed
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
    │   ├── calls self.system_monitor.log_stats  # the CPU / GPU stats onto the logger
    │   ├── impls log the iteration time via time.time
    │   └── if flush_prefix is not None
    │       └── impls flush the logger with flush_prefix
    ├── def _after_val_loop_(self) -> None
    │   ├── # Spawns a background thread that saves validation scores, updates early stopping, relinks the best checkpoint, saves debug outputs, and cleans checkpoints.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── def after_val_ops() [local]
    │   │   ├── # Saves the epoch's validation scores and best-checkpoint link and cleans old checkpoints.
    │   │   ├── impls epoch_root = work_dir's epoch_{self.cum_epochs}
    │   │   ├── impls os.makedirs the epoch root
    │   │   ├── with self.buffer_lock
    │   │   │   └── impls summarize the metric scores to validation_scores.json
    │   │   ├── impls json.load validation_scores.json back to verify the write
    │   │   ├── if self.early_stopping
    │   │   │   └── impls update early stopping with the new scores
    │   │   ├── try
    │   │   │   ├── calls self._find_best_checkpoint  # → best_checkpoint
    │   │   │   ├── impls soft_link = work_dir's checkpoint_best.pt
    │   │   │   ├── if os.path.isfile(soft_link)
    │   │   │   │   └── impls os.system rm the existing checkpoint_best.pt link
    │   │   │   └── impls os.system ln -s the best checkpoint to checkpoint_best.pt
    │   │   ├── except
    │   │   │   └── impls set best_checkpoint to None
    │   │   ├── if self.debugger and self.debugger.enabled
    │   │   │   ├── impls debugger_dir = the epoch root's debugger
    │   │   │   └── impls save the debugger outputs to it
    │   │   └── calls self._clean_checkpoints  # latest_checkpoint, best_checkpoint (may be None)
    │   ├── calls after_val_ops  # the threading.Thread target
    │   ├── impls self.after_val_thread = threading.Thread(target=after_val_ops)
    │   └── impls start the after-val thread
    ├── def _before_test_loop_(self) -> str
    │   ├── # Loads the best checkpoint into the model, sets eval mode, resets the metric buffer and sets the test seed, and returns the checkpoint path.
    │   ├── calls self._find_best_checkpoint  # → checkpoint_filepath
    │   ├── impls checkpoint = torch.load of the best checkpoint
    │   ├── impls load_state_dict it into the model
    │   ├── impls model.eval
    │   ├── impls reset the metric buffer
    │   ├── impls set the test dataloader's base seed to self.test_seed
    │   └── return  # the best checkpoint path
    ├── def _after_test_loop_(self, best_checkpoint: str) -> None
    │   ├── # Writes the test scores and best-checkpoint path to test/test_results.json.
    │   ├── if self.work_dir is None
    │   │   └── return
    │   ├── impls test_root = work_dir's test
    │   ├── impls os.makedirs the test root
    │   ├── calls serialize_tensor  # the metric summary
    │   ├── impls results = the serialized metric scores and best_checkpoint  # impls-node-one-step:skip
    │   └── impls jsbeautifier-dump the results to test_results.json
    ├── def _find_best_checkpoint(self) -> str
    │   ├── # Scans finished epochs and returns the checkpoint path with the best validation score.
    │   ├── calls get_metric_directions  # self.metric → metric_directions
    │   ├── impls order_config = self.config.get('order', False)
    │   ├── impls best_epoch_dir = None
    │   ├── impls best_scores = None
    │   ├── impls epoch_idx = 0
    │   ├── while epoch_idx < self.tot_epochs
    │   │   ├── impls epoch_dir = work_dir's epoch_{epoch_idx}
    │   │   ├── calls TrainingJob._check_epoch_finished  # epoch_dir, expected_files
    │   │   ├── if that epoch has not finished
    │   │   │   └── break
    │   │   ├── impls scores_path = the epoch dir's validation_scores.json
    │   │   ├── impls validation_scores = json.load of scores_path
    │   │   ├── impls current_scores = the 'aggregated' entry of those scores
    │   │   ├── if best_scores is None
    │   │   │   └── impls set this epoch as best
    │   │   ├── else
    │   │   │   ├── calls compare_scores  # current_scores, best_scores, order_config, metric_directions → is_better
    │   │   │   └── if is_better
    │   │   │       └── impls set this epoch as best
    │   │   └── impls advance epoch_idx by one
    │   ├── if best_epoch_dir is None
    │   │   └── raise ValueError
    │   ├── impls best_checkpoint = the best epoch dir's checkpoint.pt
    │   └── return  # the best epoch's checkpoint path
    ├── def _clean_checkpoints(self, latest_checkpoint: str, best_checkpoint: Optional[str] = None) -> None
    │   ├── # Removes epoch checkpoints outside the keep-set (latest, best, precomputed indices), in parallel.
    │   ├── impls checkpoint_method = self.config.get('checkpoint_method', 'latest')
    │   ├── if checkpoint_method == 'all'
    │   │   └── return
    │   ├── impls keep_checkpoints = [latest_checkpoint]
    │   ├── if best_checkpoint is not None
    │   │   └── impls append best_checkpoint to keep_checkpoints
    │   ├── impls extend keep_checkpoints with the checkpoint_indices epochs
    │   ├── impls existing_checkpoints = glob.glob of the epoch checkpoints under work_dir
    │   ├── def clean_single_checkpoint(checkpoint: str) -> None [local]
    │   │   ├── # Removes one checkpoint when it is outside the keep-set and its next epoch has finished.
    │   │   ├── if checkpoint in keep_checkpoints
    │   │   │   └── return
    │   │   ├── impls epoch_dir = the checkpoint's directory
    │   │   ├── impls epoch = the epoch dir's index
    │   │   ├── calls TrainingJob._check_epoch_finished  # on the NEXT epoch's dir
    │   │   └── if that next epoch has finished
    │   │       └── impls os.system rm -f the checkpoint
    │   └── with ThreadPoolExecutor() as executor
    │       └── calls clean_single_checkpoint  # executor.map over the existing checkpoints
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
