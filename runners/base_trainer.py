from typing import Tuple, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import copy
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import jsbeautifier
import torch
import criteria
import utils
from utils.builders import build_from_config
from utils.io import serialize_tensor
from utils.automation.run_status import check_epoch_finished
from utils.monitor.gpu_monitor import GPUMonitor
from utils.logging.text_logger import TextLogger
from utils.logging.screen_logger import ScreenLogger


class BaseTrainer(ABC):

    def __init__(
        self,
        config: dict,
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        r"""
        Args:
            config (dict): the config dict that controls the entire pipeline.
        """
        assert type(config) == dict, f"{type(config)=}"
        self.config = copy.deepcopy(config)
        assert type(device) == torch.device, f"{type(device)=}"
        self.device = device
        self.eval_n_jobs = self.config.get('eval_n_jobs', 1)
        self._init_work_dir()
        self._init_tot_epochs()
        torch.autograd.set_detect_anomaly(True)

    # ====================================================================================================
    # ====================================================================================================

    def _init_work_dir(self) -> None:
        if self.config.get('work_dir', None):
            work_dir = self.config['work_dir']
            assert type(work_dir) == str, f"{type(work_dir)=}"
            os.makedirs(work_dir, exist_ok=True)
            self.work_dir = work_dir
        else:
            self.work_dir = None

    def _init_tot_epochs(self) -> None:
        assert 'epochs' in self.config.keys()
        tot_epochs = self.config['epochs']
        assert type(tot_epochs) == int, f"{type(tot_epochs)=}"
        assert tot_epochs >= 0, f"{tot_epochs=}"
        self.tot_epochs = tot_epochs

    def _init_logger(self) -> None:
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
        log_filepath = os.path.join(self.work_dir, f"train_val_{session_idx}.log")

        # Try to initialize screen logger, fall back to traditional logger if it fails
        try:
            self.logger = ScreenLogger(max_iterations=10, filepath=log_filepath, layout="train")
        except Exception as e:
            print(f"Failed to initialize screen logger: {e}. Falling back to traditional logger.")
            self.logger = TextLogger(filepath=log_filepath)

        # config log
        with open(os.path.join(self.work_dir, "config.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(str(self.config), jsbeautifier.default_options()))

        # Initialize GPU monitor
        if torch.cuda.is_available():
            self.gpu_monitor = GPUMonitor()
        else:
            self.gpu_monitor = None

    def _init_determinism_(self) -> None:
        self.logger.info("Initializing determinism...")
        utils.determinism.set_determinism()

        # Get training seeds
        assert 'train_seeds' in self.config.keys()
        train_seeds = self.config['train_seeds']
        assert type(train_seeds) == list, f"{type(train_seeds)=}"
        assert all(type(seed) == int for seed in train_seeds), f"{train_seeds=}"
        assert len(train_seeds) == self.tot_epochs, f"{len(train_seeds)=}, {self.tot_epochs=}"
        self.train_seeds = train_seeds

        # Set init seed
        assert 'init_seed' in self.config.keys()
        init_seed = self.config['init_seed']
        assert type(init_seed) == int, f"{type(init_seed)=}"
        utils.determinism.set_seed(seed=init_seed)

    @property
    def expected_files(self) -> List[str]:
        return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]

    def _init_state_(self) -> None:
        self.logger.info("Initializing state...")
        # Get self.cum_epochs
        if self.work_dir is None:
            self.cum_epochs = 0
            return
        # determine where to resume from
        load_idx: Optional[int] = None
        for idx in range(self.tot_epochs):
            if not check_epoch_finished(
                epoch_dir=os.path.join(self.work_dir, f"epoch_{idx}"),
                expected_files=self.expected_files,
            ):
                break
            if os.path.isfile(os.path.join(self.work_dir, f"epoch_{idx}", "checkpoint.pt")):
                load_idx = idx
        # resume state
        if load_idx is None:
            self.logger.info("Training from scratch.")
            self.cum_epochs = 0
            return
        self.cum_epochs = load_idx + 1

    def _init_dataloaders_(self) -> None:
        self.logger.info("Initializing dataloaders...")
        # initialize training dataloader
        if self.config.get('train_dataset', None) and self.config.get('train_dataloader', None):
            train_dataset: torch.utils.data.Dataset = build_from_config(self.config['train_dataset'])
            self.train_dataloader: torch.utils.data.DataLoader = build_from_config(
                dataset=train_dataset, shuffle=True, config=self.config['train_dataloader'],
            )
        else:
            self.train_dataloader = None
        # initialize validation dataloader
        if self.config.get('val_dataset', None) and self.config.get('val_dataloader', None):
            val_dataset: torch.utils.data.Dataset = build_from_config(self.config['val_dataset'])
            if 'batch_size' not in self.config['val_dataloader']['args']:
                self.config['val_dataloader']['args']['batch_size'] = 1
            self.val_dataloader: torch.utils.data.DataLoader = build_from_config(
                dataset=val_dataset, shuffle=False, config=self.config['val_dataloader'],
            )
        else:
            self.val_dataloader = None
        # initialize test dataloader
        if self.config.get('test_dataset', None) and self.config.get('test_dataloader', None):
            test_dataset: torch.utils.data.Dataset = build_from_config(self.config['test_dataset'])
            if 'batch_size' not in self.config['test_dataloader']['args']:
                self.config['test_dataloader']['args']['batch_size'] = 1
            self.test_dataloader: torch.utils.data.DataLoader = build_from_config(
                dataset=test_dataset, shuffle=False, config=self.config['test_dataloader'],
            )
        else:
            self.test_dataloader = None

    def _init_criterion_(self) -> None:
        self.logger.info("Initializing criterion...")
        if self.config.get('criterion', None):
            criterion = build_from_config(self.config['criterion'])
            assert isinstance(criterion, criteria.BaseCriterion) and isinstance(criterion, torch.nn.Module), f"{type(criterion)=}"
            criterion = criterion.to(self.device)
            self.criterion = criterion
        else:
            self.criterion = None

    def _init_metric_(self) -> None:
        self.logger.info("Initializing metric...")
        if self.config.get('metric', None):
            self.metric = build_from_config(self.config['metric'])
        else:
            self.metric = None

    def _init_model_(self) -> None:
        self.logger.info("Initializing model...")
        if self.config.get('model', None):
            model = build_from_config(self.config['model'])
            assert isinstance(model, torch.nn.Module), f"{type(model)=}"
            model = model.to(self.device)
            self.model = model
        else:
            self.model = None

    @abstractmethod
    def _init_optimizer_(self) -> None:
        raise NotImplementedError("Abstract method BaseTrainer._init_optimizer_ not implemented.")

    @abstractmethod
    def _init_scheduler_(self) -> None:
        raise NotImplementedError("Abstract method BaseTrainer._init_scheduler_ not implemented.")

    def _load_checkpoint_(self) -> None:
        if self.cum_epochs == 0:
            return
        checkpoint_filepath = os.path.join(self.work_dir, f"epoch_{self.cum_epochs-1}", "checkpoint.pt")
        self.logger.info(f"Loading checkpoint from {checkpoint_filepath}...")
        checkpoint = torch.load(checkpoint_filepath)
        assert type(checkpoint) == dict, f"{type(checkpoint)=}"
        for component in ['model', 'optimizer', 'scheduler']:
            assert hasattr(self, component), f"{component=}"
            assert getattr(self, component) is not None, f"{component=}"
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # ====================================================================================================
    # iteration-level methods
    # ====================================================================================================

    @abstractmethod
    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        raise NotImplementedError("Abstract method BaseTrainer._set_gradients_ not implemented .")

    def _train_step(self, dp: Dict[str, Dict[str, Any]]) -> None:
        r"""
        Args:
            dp (Dict[str, Dict[str, Any]]): a dictionary containing inputs, labels, and meta info.
        """
        # init time
        start_time = time.time()

        # Start GPU monitoring if available
        if self.gpu_monitor is not None:
            self.gpu_monitor.start()

        # do computation
        dp['outputs'] = self.model(dp['inputs'])
        dp['losses'] = self.criterion(y_pred=dp['outputs'], y_true=dp['labels'])

        # update logger
        self.logger.update_buffer({"learning_rate": self.scheduler.get_last_lr()})
        self.logger.update_buffer(utils.logging.log_losses(losses=dp['losses']))

        # update states
        self._set_gradients_(dp)
        self.optimizer.step()
        self.scheduler.step()

        # Log GPU stats if available
        if self.gpu_monitor is not None:
            self.gpu_monitor.update()
            # Pass the logger to log_stats
            self.gpu_monitor.log_stats(self.logger)

        # log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    def _eval_step(self, dp: Dict[str, Dict[str, Any]], flush_prefix: Optional[str] = None) -> None:
        r"""
        Args:
            dp (Dict[str, Dict[str, Any]]): a dictionary containing inputs, labels, and meta info.
            flush_prefix (Optional[str]): the prefix to use for the flush.
        """
        # init time
        start_time = time.time()

        # Start GPU monitoring if available
        if self.gpu_monitor is not None:
            self.gpu_monitor.start()

        # do computation
        dp['outputs'] = self.model(dp['inputs'])
        dp['scores'] = self.metric(y_pred=dp['outputs'], y_true=dp['labels'])

        # Log scores
        self.logger.update_buffer(utils.logging.log_scores(scores=dp['scores']))

        # Log GPU stats if available
        if self.gpu_monitor is not None:
            self.gpu_monitor.update()
            # Pass the logger to log_stats
            self.gpu_monitor.log_stats(self.logger)

        # Log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

        # Log progress if flush_prefix is provided
        if flush_prefix is not None:
            self.logger.flush(prefix=flush_prefix)

    # ====================================================================================================
    # training and validation epochs
    # ====================================================================================================

    def _train_epoch_(self) -> None:
        if not (self.train_dataloader and self.model):
            self.logger.info("Skipped training epoch.")
            return
        if os.path.isfile(os.path.join(self.work_dir, f"epoch_{self.cum_epochs}", "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(self.work_dir, f"epoch_{self.cum_epochs}", "checkpoint.pt"))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info(f"Found trained checkpoint at {os.path.join(self.work_dir, f'epoch_{self.cum_epochs}', 'checkpoint.pt')}.")
            self.logger.info(f"Skipping training epoch {self.cum_epochs}.")
            return
        # init time
        start_time = time.time()
        # do training loop
        self.model.train()
        self.criterion.reset_buffer()
        self.optimizer.reset_buffer()
        self.logger.train()

        for idx, dp in enumerate(self.train_dataloader):
            self._train_step(dp=dp)
            self.logger.flush(prefix=f"Training [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.train_dataloader)}].")
        # after training loop
        self._after_train_loop_()
        # log time
        self.logger.info(f"Training epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _save_checkpoint_(self, output_path: str) -> None:
        r"""Default checkpoint saving method. Override to save more.

        Args:
            output_path (str): the file path to which the checkpoint will be saved.
        """
        torch.save(obj={
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, f=output_path)

    def _after_train_loop_(self) -> None:
        if self.work_dir is None:
            return
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        os.makedirs(epoch_root, exist_ok=True)
        # save criterion buffer to disk
        self.criterion.summarize(output_path=os.path.join(epoch_root, "training_losses.pt"))
        _ = torch.load(os.path.join(epoch_root, "training_losses.pt"))
        # save optimizer buffer to disk
        self.optimizer.summarize(output_path=os.path.join(epoch_root, "optimizer_buffer.json"))
        with open(os.path.join(epoch_root, "optimizer_buffer.json"), mode='r') as f:
            _ = json.load(f)
        # save checkpoint to disk
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        self._save_checkpoint_(output_path=latest_checkpoint)
        # set latest checkpoint
        soft_link: str = os.path.join(self.work_dir, "checkpoint_latest.pt")
        if os.path.islink(soft_link):
            os.system(' '.join(["rm", soft_link]))
        os.system(' '.join(["ln", "-s", os.path.relpath(path=latest_checkpoint, start=self.work_dir), soft_link]))

    def _val_epoch_(self) -> None:
        if not (self.val_dataloader and self.model):
            self.logger.info("Skipped validation epoch.")
            return
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()
        self.logger.eval()

        if self.eval_n_jobs == 1:
            self.logger.info("Running validation sequentially...")
            for idx, dp in enumerate(self.val_dataloader):
                self._eval_step(dp, flush_prefix=f"Validation [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.val_dataloader)}].")
        else:
            self.logger.info(f"Using {self.eval_n_jobs} threads for parallel validation")
            with ThreadPoolExecutor(max_workers=self.eval_n_jobs) as executor:
                future_to_args = {executor.submit(
                    self._eval_step, dp,
                    flush_prefix=f"Validation [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(self.val_dataloader)}].",
                ): (idx, dp) for idx, dp in enumerate(self.val_dataloader)}
                for future in as_completed(future_to_args):
                    future.result()

        # after validation loop
        self._after_val_loop_()
        # log time
        self.logger.info(f"Validation epoch time: {round(time.time() - start_time, 2)} seconds.")

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
        if self.work_dir is None:
            return
        # initialize epoch root directory
        epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
        os.makedirs(epoch_root, exist_ok=True)
        # save validation scores to disk
        self.metric.summarize(output_path=os.path.join(epoch_root, "validation_scores.json"))
        with open(os.path.join(epoch_root, "validation_scores.json"), mode='r') as f:
            _ = json.load(f)
        # set best checkpoint
        try:
            best_checkpoint: str = self._find_best_checkpoint_()
            soft_link: str = os.path.join(self.work_dir, "checkpoint_best.pt")
            if os.path.isfile(soft_link):
                os.system(' '.join(["rm", soft_link]))
            os.system(' '.join(["ln", "-s", os.path.relpath(path=best_checkpoint, start=self.work_dir), soft_link]))
        except:
            best_checkpoint = None
        # cleanup checkpoints
        checkpoints: List[str] = glob.glob(os.path.join(self.work_dir, "epoch_*", "checkpoint.pt"))
        if best_checkpoint is not None:
            checkpoints.remove(best_checkpoint)
        latest_checkpoint = os.path.join(epoch_root, "checkpoint.pt")
        if latest_checkpoint in checkpoints:
            checkpoints.remove(latest_checkpoint)

        # Handle different checkpoint methods
        checkpoint_method = self.config.get('checkpoint_method', 'latest')
        if checkpoint_method == 'all':
            # Keep all checkpoints
            return
        elif checkpoint_method == 'latest':
            # Keep only the latest checkpoint
            for checkpoint in checkpoints:
                assert checkpoint.endswith("checkpoint.pt")
                epoch_dir = os.path.dirname(checkpoint)
                assert os.path.basename(epoch_dir).startswith("epoch_")
                epoch = int(os.path.basename(epoch_dir).split('_')[1])
                # remove only if next epoch has finished
                if check_epoch_finished(
                    epoch_dir=os.path.join(os.path.dirname(epoch_dir), f"epoch_{epoch+1}"),
                    expected_files=self.expected_files,
                ):
                    os.system(' '.join(["rm", "-f", checkpoint]))
        else:
            # Handle interval-based checkpointing
            assert isinstance(checkpoint_method, int), "checkpoint_method must be 'all', 'latest', or a positive integer"
            assert checkpoint_method > 0, "checkpoint_method interval must be positive"

            for checkpoint in checkpoints:
                assert checkpoint.endswith("checkpoint.pt")
                epoch_dir = os.path.dirname(checkpoint)
                assert os.path.basename(epoch_dir).startswith("epoch_")
                epoch = int(os.path.basename(epoch_dir).split('_')[1])

                # Keep checkpoints at the specified interval
                if epoch % checkpoint_method == checkpoint_method - 1:
                    continue

                # remove only if next epoch has finished
                if check_epoch_finished(
                    epoch_dir=os.path.join(os.path.dirname(epoch_dir), f"epoch_{epoch+1}"),
                    expected_files=self.expected_files,
                ):
                    os.system(' '.join(["rm", "-f", checkpoint]))

    # ====================================================================================================
    # test epoch
    # ====================================================================================================

    @torch.no_grad()
    def _test_epoch_(self) -> None:
        if not (self.test_dataloader and self.model):
            self.logger.info("Skipped test epoch.")
            return
        # init time
        start_time = time.time()
        # before test loop
        best_checkpoint: str = self._before_test_loop_()
        # do test loop
        self.model.eval()
        self.metric.reset_buffer()
        for idx, dp in enumerate(self.test_dataloader):
            self._eval_step(dp=dp, flush_prefix=f"Test epoch [Iteration {idx}/{len(self.test_dataloader)}].")
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
        if self.work_dir is None:
            return
        # initialize test results directory
        test_root = os.path.join(self.work_dir, "test")
        os.makedirs(test_root, exist_ok=True)
        # save test results to disk
        results = {
            'scores': serialize_tensor(self.metric.summarize()),
            'checkpoint_filepath': best_checkpoint,
        }
        with open(os.path.join(test_root, "test_results.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(json.dumps(results), jsbeautifier.default_options()))

    # ====================================================================================================
    # ====================================================================================================

    def _init_components_(self):
        self._init_logger()
        self._init_determinism_()
        self._init_state_()
        self._init_dataloaders_()
        self._init_criterion_()
        self._init_metric_()
        self._init_model_()
        self._init_optimizer_()
        self._init_scheduler_()
        self._load_checkpoint_()

    def run(self):
        # initialize run
        self._init_components_()
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
