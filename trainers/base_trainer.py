from typing import Tuple, List, Dict, Any
from abc import abstractmethod
import os
import glob
import time
import json
import jsbeautifier
import torch
import wandb

import criteria
import utils
from utils.builder import build_from_config
from utils.ops import apply_tensor_op
from utils.io import serialize_tensor

try:
    # torch 2.x
    from torch.optim.lr_scheduler import LRScheduler
except:
    # torch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class BaseTrainer:

    def __init__(self, config: dict):
        self.config = config
        # init work dir
        assert 'work_dir' in self.config.keys()
        assert type(self.config['work_dir']) == str, f"{type(self.config['work_dir'])=}"
        work_dir = self.config['work_dir']
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        # init total epochs
        assert 'epochs' in self.config.keys()
        self.tot_epochs = self.config['epochs']

    # ====================================================================================================
    # ====================================================================================================

    def _init_logger_(self):
        session_idx: int = len(glob.glob(os.path.join(self.work_dir, "train_val*.log")))
        # git log
        git_log = os.path.join(self.work_dir, f"git_{session_idx}.log")
        utils.logging.echo_page_break(filepath=git_log, heading="git branch -a")
        os.system(f"git branch -a >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git status")
        os.system(f"git status >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git log")
        os.system(f"git log >> {git_log}")
        # training log
        self.logger = utils.logging.Logger(
            filepath=os.path.join(self.work_dir, f"train_val_{session_idx}.log"),
        )

    def _init_determinism_(self):
        self.logger.info("Initializing determinism...")
        utils.determinism.set_determinism()
        # get seed for initialization steps
        assert 'init_seed' in self.config.keys()
        init_seed = self.config['init_seed']
        assert type(init_seed) == int, f"{type(init_seed)=}"
        utils.determinism.set_seed(seed=init_seed)
        # get seeds for training
        assert 'train_seeds' in self.config.keys()
        train_seeds = self.config['train_seeds']
        assert type(train_seeds) == list, f"{type(train_seeds)=}"
        for seed in train_seeds:
            assert type(seed) == int, f"{type(seed)=}"
        assert len(train_seeds) == self.tot_epochs, f"{len(train_seeds)=}, {self.tot_epochs=}"
        self.train_seeds = train_seeds

    def _init_dataloaders_(self):
        self.logger.info("Initializing dataloaders...")
        # initialize training dataloader
        assert 'train_dataset' in self.config and 'train_dataloader' in self.config
        train_dataset: torch.utils.data.Dataset = build_from_config(self.config['train_dataset'])
        if 'collate_fn' in self.config['train_dataloader']['args']:
            self.config['train_dataloader']['args']['collate_fn'] = build_from_config(self.config['train_dataloader']['args']['collate_fn'])
        self.train_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=train_dataset, shuffle=True, config=self.config['train_dataloader'],
        )
        # initialize validation dataloader
        assert 'val_dataset' in self.config and 'val_dataloader' in self.config
        val_dataset: torch.utils.data.Dataset = build_from_config(self.config['val_dataset'])
        if 'collate_fn' in self.config['val_dataloader']['args']:
            self.config['val_dataloader']['args']['collate_fn'] = build_from_config(self.config['val_dataloader']['args']['collate_fn'])
        self.val_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=val_dataset, shuffle=False, batch_size=1, config=self.config['val_dataloader'],
        )
        # initialize test dataloader
        assert 'test_dataset' in self.config and 'test_dataloader' in self.config
        test_dataset: torch.utils.data.Dataset = build_from_config(self.config['test_dataset'])
        if 'collate_fn' in self.config['test_dataloader']['args']:
            self.config['test_dataloader']['args']['collate_fn'] = build_from_config(self.config['test_dataloader']['args']['collate_fn'])
        self.test_dataloader: torch.utils.data.DataLoader = build_from_config(
            dataset=test_dataset, shuffle=False, batch_size=1, config=self.config['test_dataloader'],
        )

    def _init_model_(self):
        self.logger.info("Initializing model...")
        assert 'model' in self.config
        model = build_from_config(self.config['model'])
        assert isinstance(model, torch.nn.Module), f"{type(model)=}"
        model = model.cuda()
        self.model = model

    def _init_criterion_(self):
        self.logger.info("Initializing criterion...")
        assert 'criterion' in self.config
        criterion = build_from_config(self.config['criterion'])
        assert isinstance(criterion, criteria.BaseCriterion) and isinstance(criterion, torch.nn.Module), f"{type(criterion)=}"
        criterion = criterion.cuda()
        self.criterion = criterion

    def _init_metric_(self):
        self.logger.info("Initializing metric...")
        assert 'metric' in self.config
        self.metric = build_from_config(self.config['metric'])

    def _init_optimizer_(self):
        r"""Requires self.model.
        """
        self.logger.info("Initializing optimizer...")
        assert 'optimizer' in self.config
        assert hasattr(self, 'model') and isinstance(self.model, torch.nn.Module)
        self.optimizer: torch.optim.Optimizer = build_from_config(
            params=self.model.parameters(), config=self.config['optimizer'],
        )

    def _init_scheduler_(self):
        r"""Requires self.train_dataloader and self.optimizer.
        """
        self.logger.info("Initializing scheduler...")
        assert 'scheduler' in self.config
        assert hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        assert hasattr(self, 'optimizer') and isinstance(self.optimizer, torch.optim.Optimizer)
        self.config['scheduler']['args']['lr_lambda'] = build_from_config(
            steps=len(self.train_dataloader), config=self.config['scheduler']['args']['lr_lambda'],
        )
        self.scheduler: LRScheduler = build_from_config(
            optimizer=self.optimizer, config=self.config['scheduler'],
        )

    @property
    def expected_files(self):
        return ["training_losses.pt", "validation_scores.json"]

    def _load_checkpoint_(self, checkpoint: dict) -> None:
        r"""Default checkpoint loading method. Override to load more.

        Args:
            checkpoint (dict): the output of torch.load(checkpoint_filepath).
        """
        assert type(checkpoint) == dict, f"{type(checkpoint)=}"
        self.cum_epochs = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def _init_state_(self):
        self.logger.info("Initializing state...")
        # init epoch numbers
        self.cum_epochs = 0
        # determine where to resume from
        load_idx: int = None
        for idx in range(self.tot_epochs):
            epoch_finished = all([os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", filename))
                for filename in self.expected_files
            ])
            if not epoch_finished:
                break
            if os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", "checkpoint.pt")):
                load_idx = idx
        if load_idx is None:
            self.logger.info("Training from scratch.")
            return
        # resume state
        checkpoint_filepath = os.path.join(self.work_dir, f"epoch_{load_idx}", "checkpoint.pt")
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_filepath}...")
            checkpoint = torch.load(checkpoint_filepath)
            self._load_checkpoint_(checkpoint)
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to load checkpoint at {checkpoint_filepath}: {e}")

    # ====================================================================================================
    # iteration-level methods
    # ====================================================================================================

    @abstractmethod
    def _set_gradients_(self, example: dict):
        raise NotImplementedError("[ERROR] _set_gradients_ not implemented for base class.")

    def _train_step_(self, example: dict) -> None:
        r"""
        Args:
            example (dict): a dictionary containing inputs and ground-truth labels.
        """
        # init time
        start_time = time.time()
        # copy to GPU
        example = apply_tensor_op(func=lambda x: x.cuda(), inputs=example)
        # do computation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            example['outputs'] = self.model(example['inputs'])
            example['losses'] = self.criterion(y_pred=example['outputs'], y_true=example['labels'])
        # update logger
        self.logger.update_buffer({"learning_rate": self.scheduler.get_last_lr()})
        self.logger.update_buffer(utils.logging.log_losses(example['losses']))
        # update states
        self._set_gradients_(example)
        self.optimizer.step()
        self.scheduler.step()
        # log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    def _eval_step_(self, example: dict) -> None:
        r"""
        Args:
            example (dict): a dictionary containing inputs and ground-truth labels.
        """
        # init time
        start_time = time.time()
        # copy to GPU
        example = apply_tensor_op(func=lambda x: x.cuda(), inputs=example)
        # do computation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            example['outputs'] = self.model(example['inputs'])
            example['scores'] = self.metric(y_pred=example['outputs'], y_true=example['labels'])
        # update logger
        self.logger.update_buffer(utils.logging.log_scores(example['scores']))
        # log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    # ====================================================================================================
    # training and validation epochs
    # ====================================================================================================

    def _train_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # do training loop
        self.model.train()
        self.criterion.reset_buffer()
        for idx, example in enumerate(self.train_dataloader):
            self._train_step_(example=example)
            self.logger.flush(prefix=f"Training [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.train_dataloader)}].")
        # after training loop
        self._after_train_loop_()
        # log time
        self.logger.info(f"Training epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _val_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()
        for idx, example in enumerate(self.val_dataloader):
            self._eval_step_(example=example)
            self.logger.flush(prefix=f"Validation [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.val_dataloader)}].")
        # after validation loop
        self._after_val_loop_()
        # log time
        self.logger.info(f"Validation epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _save_checkpoint_(self, output_path: str) -> None:
        r"""Default checkpoint saving method. Override to save more.

        Args:
            output_path (str): the file path to which the checkpoint will be saved.
        """
        torch.save(obj={
            'epoch': self.cum_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, f=output_path)

    def _after_train_loop_(self) -> None:
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        if not os.path.isdir(epoch_root):
            os.makedirs(epoch_root)
        # save training losses to disk
        self.criterion.summarize(output_path=os.path.join(epoch_root, "training_losses.pt"))
        # save checkpoint to disk
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        self._save_checkpoint_(output_path=latest_checkpoint)
        # set latest checkpoint
        soft_link: str = os.path.join(self.work_dir, "checkpoint_latest.pt")
        if os.path.isfile(soft_link):
            os.system(' '.join(["rm", soft_link]))
        os.system(' '.join(["ln", "-s", os.path.relpath(path=latest_checkpoint, start=self.work_dir), soft_link]))

    def _find_best_checkpoint_(self) -> str:
        r"""
        Returns:
            best_checkpoint (str): the filepath to the checkpoint with the highest validation score.
        """
        avg_scores: List[Tuple[str, Any]] = []
        for epoch_dir in sorted(glob.glob(os.path.join(self.work_dir, "epoch_*"))):
            with open(os.path.join(epoch_dir, "validation_scores.json"), mode='r') as f:
                scores: Dict[str, float] = json.load(f)
            avg_scores.append((epoch_dir, scores))
        best_epoch_dir: str = max(avg_scores, key=lambda x: x[1]['reduced'])[0]
        best_checkpoint: str = os.path.join(best_epoch_dir, "checkpoint.pt")
        assert os.path.isfile(best_checkpoint), f"{best_checkpoint=}"
        return best_checkpoint

    def _after_val_loop_(self) -> None:
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        if not os.path.isdir(epoch_root):
            os.makedirs(epoch_root)
        # save validation scores to disk
        self.metric.summarize(output_path=os.path.join(epoch_root, "validation_scores.json"))
        # set best checkpoint
        best_checkpoint: str = self._find_best_checkpoint_()
        soft_link: str = os.path.join(self.work_dir, "checkpoint_best.pt")
        if os.path.isfile(soft_link):
            os.system(' '.join(["rm", soft_link]))
        os.system(' '.join(["ln", "-s", os.path.relpath(path=best_checkpoint, start=self.work_dir), soft_link]))
        # cleanup checkpoints
        checkpoints: List[str] = glob.glob(os.path.join(self.work_dir, "epoch_*", "checkpoint.pt"))
        checkpoints.remove(best_checkpoint)
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        if latest_checkpoint in checkpoints:
            checkpoints.remove(latest_checkpoint)
        os.system(' '.join(["rm", "-f"] + checkpoints))

    # ====================================================================================================
    # test epoch
    # ====================================================================================================

    @torch.no_grad()
    def _test_epoch_(self) -> None:
        # init time
        start_time = time.time()
        # before test loop
        best_checkpoint: str = self._before_test_loop_()
        # do test loop
        self.model.eval()
        self.metric.reset_buffer()
        for idx, example in enumerate(self.test_dataloader):
            self._eval_step_(example=example)
            self.logger.flush(prefix=f"Test epoch [Iteration {idx}/{len(self.test_dataloader)}].")
        # after test loop
        self._after_test_loop_(best_checkpoint=best_checkpoint)
        # log time
        self.logger.info(f"Test epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _before_test_loop_(self) -> str:
        checkpoint_filepath = self._find_best_checkpoint_()
        checkpoint = torch.load(checkpoint_filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint_filepath

    def _after_test_loop_(self, best_checkpoint: str) -> None:
        # initialize test results directory
        test_root = os.path.join(self.work_dir, "test")
        if not os.path.isdir(test_root):
            os.makedirs(test_root)
        # save test results to disk
        results = {
            'scores': serialize_tensor(self.metric.summarize()),
            'checkpoint_filepath': best_checkpoint,
        }
        with open(os.path.join(test_root, "test_results.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(json.dumps(results), jsbeautifier.default_options()))

    # ====================================================================================================
    # ====================================================================================================

    def is_running(self, wait_time: float) -> bool:
        logs = glob.glob(os.path.join(self.work_dir, "train_val*.log"))
        if len(logs) == 0:
            return False
        return time.time() - max([os.path.getmtime(fp) for fp in logs]) <= wait_time

    def has_finished(self) -> bool:
        # check train/val
        for idx in range(self.tot_epochs):
            epoch_finished = all([
                os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", filename))
                for filename in self.expected_files
            ])
            if not epoch_finished:
                return False
        # check test
        if not os.path.isfile(os.path.join(self.work_dir, "test", "test_results.json")):
            return False
        return True

    def has_failed(self, wait_time: float) -> bool:
        return not self.is_running(wait_time=wait_time) and not self.has_finished()

    # ====================================================================================================
    # ====================================================================================================

    def _init_components_(self):
        self._init_logger_()
        self._init_determinism_()
        self._init_dataloaders_()
        self._init_model_()
        self._init_criterion_()
        self._init_metric_()
        self._init_optimizer_()
        self._init_scheduler_()
        self._init_state_()

    def train(self):
        # skip if finished
        if self.has_finished():
            print("Session already finished.")
            return
        # initialize components
        self._init_components_()
        # initialize run
        wandb.init()
        start_epoch = self.cum_epochs
        self.logger.page_break()
        # training and validation epochs
        for idx in range(start_epoch, self.tot_epochs):
            utils.determinism.set_seed(seed=self.train_seeds[idx])
            self._train_epoch_()
            self._val_epoch_()
            self.logger.page_break()
            self.cum_epochs = idx + 1
        # test epoch
        self._test_epoch_()
        # cleanup run
        wandb.finish()
