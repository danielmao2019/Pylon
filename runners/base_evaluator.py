from typing import List, Optional
import copy
import os
import glob
import time
import json
import jsbeautifier
import torch
import concurrent.futures

import utils
from utils.builders import build_from_config


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

    def _process_eval_batch(self, batch_data):
        """Process a single evaluation batch in a thread-safe manner."""
        # Run model inference
        batch_data['outputs'] = self.model(batch_data['inputs'])
        batch_data['scores'] = self.metric(y_pred=batch_data['outputs'], y_true=batch_data['labels'])
        # Add scores to the metric buffer in a thread-safe way
        self.metric.add_to_buffer(batch_data['scores'])

        # Update logger with scores
        self.logger.update_buffer(utils.logging.log_scores(scores=batch_data['scores']))

        return batch_data

    def _eval_epoch_(self) -> None:
        assert self.eval_dataloader and self.model
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()

        # Process evaluation data in parallel using threads
        # Get the number of CPU cores to use (leave one core free for system processes)
        num_workers = max(1, os.cpu_count() - 1)
        self.logger.info(f"Using {num_workers} threads for parallel evaluation")

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches to the executor
            future_to_idx = {executor.submit(self._process_eval_batch, dp): idx
                            for idx, dp in enumerate(self.eval_dataloader)}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                _ = future.result()
                self.logger.flush(prefix=f"Evaluation [Iteration {idx}/{len(self.eval_dataloader)}].")

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
