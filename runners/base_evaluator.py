from typing import List, Optional, Tuple, Any
import copy
import os
import glob
import time
import json
import jsbeautifier
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import utils
from utils.builders import build_from_config
from utils.parallelism import parallel_execute


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
        if self.work_dir is None:
            self.logger = utils.logging.Logger(filepath=None)
            return
        session_idx: int = len(glob.glob(os.path.join(self.work_dir, "eval*.log")))
        # git log
        git_log = os.path.join(self.work_dir, f"git_{session_idx}.log")
        utils.logging.echo_page_break(filepath=git_log, heading="git branch -a")
        os.system(f"git branch -a >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git status")
        os.system(f"git status >> {git_log}")
        utils.logging.echo_page_break(filepath=git_log, heading="git log")
        os.system(f"git log >> {git_log}")
        # evaluation log
        self.logger = utils.logging.Logger(
            filepath=os.path.join(self.work_dir, f"eval_{session_idx}.log"),
        )
        # config log
        with open(os.path.join(self.work_dir, "config.json"), mode='w') as f:
            f.write(jsbeautifier.beautify(str(self.config), jsbeautifier.default_options()))

    def _init_determinism_(self) -> None:
        self.logger.info("Initializing determinism...")
        utils.determinism.set_determinism()
        # get seed for initialization steps
        assert 'seed' in self.config.keys()
        seed = self.config['seed']
        assert type(seed) == int, f"{type(seed)=}"
        utils.determinism.set_seed(seed=seed)

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

    def _eval_epoch_(self, epoch: int) -> None:
        """
        Process one evaluation epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        self.metric.reset_buffer()  # Reset metric buffer before evaluation
        
        # Prepare arguments for parallel processing
        eval_args = []
        for idx, data_point in enumerate(self.eval_dataloader):
            eval_args.append((idx, data_point, self.model, self.metric, self.device))
        
        # Process batches in parallel
        with mp.Pool(processes=self.eval_n_jobs) as pool:
            results = pool.map(_worker_process_eval_batch, eval_args)
        
        # Check for errors
        errors = [r for r in results if r is not None]
        if errors:
            self.logger.error(f"Encountered {len(errors)} errors during evaluation")
            for error in errors:
                self.logger.error(f"Evaluation error: {str(error)}")
        
        # Log completion
        self.logger.info(f"Completed evaluation epoch {epoch}")

    def _after_eval_loop_(self) -> None:
        if self.work_dir is None:
            return
        # initialize test results directory
        test_root = os.path.join(self.work_dir, "test")
        os.makedirs(test_root, exist_ok=True)
        
        # Aggregate results from all worker processes
        # The metric's summarize method will handle the aggregation of the buffer
        self.metric.summarize(output_path=os.path.join(test_root, "evaluation_scores.json"))

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
        self._eval_epoch_(0)

# Update the worker function to not use global variables
def _worker_process_eval_batch(args: tuple) -> Optional[Exception]:
    """
    Worker function to process a single evaluation batch.
    
    Args:
        args: Tuple containing (index, data_point, model, metric, device)
        
    Returns:
        None if successful, Exception if an error occurred
    """
    try:
        idx, data_point, model, metric, device = args
        
        # Move data to device
        if isinstance(data_point, (tuple, list)):
            data = data_point[0].to(device)
            labels = data_point[1]
        else:
            data = data_point.to(device)
            labels = None
            
        # Forward pass
        with torch.no_grad():
            outputs = model(data)
            
        # Update metric directly using shared buffer
        metric(y_pred=outputs, y_true=labels)
        
        return None
        
    except Exception as e:
        return e
