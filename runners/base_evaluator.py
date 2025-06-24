from typing import List, Optional, Dict, Any
import copy
import os
import glob
import time
import json
import jsbeautifier
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.builders import build_from_config
from utils.determinism import set_determinism, set_seed
from utils.monitor.system_monitor import SystemMonitor
from utils.monitor.adaptive_executor import create_adaptive_executor
from utils.logging.text_logger import TextLogger
from utils.logging import echo_page_break, log_scores


class BaseEvaluator:

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

    def _init_logger(self) -> None:
        session_idx: int = len(glob.glob(os.path.join(self.work_dir, "eval*.log")))
        # git log
        git_log = os.path.join(self.work_dir, f"git_{session_idx}.log")
        echo_page_break(filepath=git_log, heading="git branch -a")
        os.system(f"git branch -a >> {git_log}")
        echo_page_break(filepath=git_log, heading="git status")
        os.system(f"git status >> {git_log}")
        echo_page_break(filepath=git_log, heading="git log")
        os.system(f"git log >> {git_log}")

        # evaluation log
        log_filepath = os.path.join(self.work_dir, f"eval_{session_idx}.log")
        self.logger = TextLogger(
            filepath=log_filepath
        )
        # config log
        with open(os.path.join(self.work_dir, "config.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(str(self.config), jsbeautifier.default_options()))

        # Initialize system monitor (CPU + GPU)
        self.system_monitor = SystemMonitor()
        self.system_monitor.start()

    def _init_determinism_(self) -> None:
        self.logger.info("Initializing determinism...")
        set_determinism()
        # get seed for initialization steps
        assert 'seed' in self.config.keys()
        seed = self.config['seed']
        assert type(seed) == int, f"{type(seed)=}"
        set_seed(seed=seed)

    def _init_dataloaders_(self) -> None:
        self.logger.info("Initializing dataloaders...")
        # initialize validation dataloader
        if self.config.get('eval_dataset', None) and self.config.get('eval_dataloader', None):
            eval_dataset: torch.utils.data.Dataset = build_from_config(self.config['eval_dataset'])
            if 'batch_size' not in self.config['eval_dataloader']['args']:
                self.config['eval_dataloader']['args']['batch_size'] = 1
            self.eval_dataloader: torch.utils.data.DataLoader = build_from_config(
                dataset=eval_dataset, shuffle=False, config=self.config['eval_dataloader'],
            )
        else:
            self.eval_dataloader = None

    def _init_model_(self) -> None:
        self.logger.info("Initializing model...")
        if self.config.get('model', None):
            model = build_from_config(self.config['model'])
            assert isinstance(model, torch.nn.Module), f"{type(model)=}"
            model = model.to(self.device)
            self.model = model
        else:
            self.model = None

    def _init_metric_(self) -> None:
        self.logger.info("Initializing metric...")
        if self.config.get('metric', None):
            self.metric = build_from_config(self.config['metric'])
        else:
            self.metric = None

    @property
    def expected_files(self) -> List[str]:
        return ["evaluation_scores.json"]

    # ====================================================================================================
    # iteration-level methods
    # ====================================================================================================

    def _eval_step(self, dp: Dict[str, Dict[str, Any]], flush_prefix: Optional[str] = None):
        """
        Args:
            dp (Dict[str, Dict[str, Any]]): a dictionary containing the batch data.
            flush_prefix (Optional[str]): the prefix to flush the logger with.
        """
        # init time
        start_time = time.time()

        # Run model inference
        dp['outputs'] = self.model(dp['inputs'])
        dp['scores'] = self.metric(y_pred=dp['outputs'], y_true=dp['labels'])

        # Log scores
        self.logger.update_buffer(log_scores(scores=dp['scores']))

        # Log system stats (CPU + GPU)
        self.system_monitor.log_stats(self.logger)

        # Log time
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

        # Log progress if flush_prefix is provided
        if flush_prefix is not None:
            self.logger.flush(prefix=flush_prefix)

    def _eval_epoch_(self) -> None:
        assert self.eval_dataloader is not None, f"{self.eval_dataloader=}"
        assert self.model is not None, f"{self.model=}"
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()
        self.logger.eval()

        if self.eval_n_jobs == 1:
            self.logger.info("Running evaluation sequentially...")
            for idx, dp in enumerate(self.eval_dataloader):
                self._eval_step(dp, flush_prefix=f"Evaluation [Iteration {idx}/{len(self.eval_dataloader)}].")
        else:
            # Use adaptive executor that dynamically adjusts worker count based on system resources
            max_workers = self.eval_n_jobs if self.eval_n_jobs > 1 else None
            executor = create_adaptive_executor(max_workers=max_workers, min_workers=1)
            self.logger.info(f"Using adaptive parallel evaluation (max {executor._max_workers} workers)")
            
            with executor:
                future_to_args = {executor.submit(
                    self._eval_step, dp,
                    flush_prefix=f"Evaluation [Iteration {idx}/{len(self.eval_dataloader)}].",
                ): (idx, dp) for idx, dp in enumerate(self.eval_dataloader)}
                for future in as_completed(future_to_args):
                    future.result()

        # after validation loop
        self._after_eval_loop_()
        # log time
        self.logger.info(f"Evaluation time: {round(time.time() - start_time, 2)} seconds.")

    def _after_eval_loop_(self) -> None:
        if self.work_dir is None:
            return
        # save validation scores to disk
        self.metric.summarize(output_path=os.path.join(self.work_dir, "evaluation_scores.json"))
        with open(os.path.join(self.work_dir, "evaluation_scores.json"), mode='r') as f:
            _ = json.load(f)

    # ====================================================================================================
    # ====================================================================================================

    def _init_components_(self):
        self._init_logger()
        self._init_determinism_()
        self._init_dataloaders_()
        self._init_model_()
        self._init_metric_()

    def run(self):
        # initialize run
        self._init_components_()
        self._eval_epoch_()
