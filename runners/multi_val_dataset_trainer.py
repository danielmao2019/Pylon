from typing import List, Optional
import os
import glob
import time
import torch
import threading
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from utils.builders import build_from_config
from utils.io import save_json
from utils.automation.run_status import check_epoch_finished


class MultiValDatasetTrainer(SupervisedSingleTaskTrainer):

    def _init_dataloaders_(self):
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
        if self.config.get('val_datasets', None) and self.config.get('val_dataloaders', None):
            assert type(self.config['val_datasets']) == list
            assert type(self.config['val_dataloaders']) == list
            assert len(self.config['val_datasets']) == len(self.config['val_dataloaders'])
            self.val_dataloaders = []
            for val_dataset_cfg, val_dataloader_cfg in zip(self.config['val_datasets'], self.config['val_dataloaders']):
                val_dataset: torch.utils.data.Dataset = build_from_config(val_dataset_cfg)
                if 'batch_size' not in val_dataloader_cfg['args']:
                    val_dataloader_cfg['args']['batch_size'] = 1
                val_dataloader: torch.utils.data.DataLoader = build_from_config(
                    dataset=val_dataset, shuffle=False, config=val_dataloader_cfg,
                )
                self.val_dataloaders.append(val_dataloader)
        else:
            self.val_dataloaders = None
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

    def _val_epoch_(self) -> None:
        if not (self.val_dataloaders and self.model):
            self.logger.info("Skipped validation epoch.")
            return

        # Wait for previous after-val operations to complete
        if self.after_val_thread and self.after_val_thread.is_alive():
            self.after_val_thread.join()

        # init time
        start_time = time.time()
        self._before_val_loop_()

        # validation loop
        results = {}
        for val_dataloader in self.val_dataloaders:
            self.metric.reset_buffer()
            for idx, dp in enumerate(val_dataloader):
                self._eval_step_(dp=dp)
                self.logger.flush(prefix=f"Validation on {val_dataloader.dataset.__class__.__name__} [Epoch {self.cum_epochs}/{self.tot_epochs}][Iteration {idx}/{len(val_dataloader)}].")
            results[val_dataloader.dataset.__class__.__name__] = self.metric.summarize(output_path=None)

        # after validation loop
        self._after_val_loop_(results)
        self.logger.info(f"Validation epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _before_val_loop_(self) -> None:
        self.model.eval()
        self.logger.eval()
        for val_dataloader in self.val_dataloaders:
            val_dataloader.dataset.set_base_seed(self.val_seeds[self.cum_epochs])

    def _after_val_loop_(self, results: dict) -> None:
        if self.work_dir is None:
            return

        def after_val_ops():
            # initialize epoch root directory
            epoch_root: str = os.path.join(self.work_dir, f"epoch_{self.cum_epochs}")
            os.makedirs(epoch_root, exist_ok=True)

            # save validation scores to disk
            save_json(obj=results, filepath=os.path.join(epoch_root, "validation_scores.json"))

            # set best checkpoint
            try:
                best_checkpoint: str = self._find_best_checkpoint_()
                soft_link: str = os.path.join(self.work_dir, "checkpoint_best.pt")
                if os.path.isfile(soft_link):
                    os.system(' '.join(["rm", soft_link]))
                os.system(' '.join(["ln", "-s", os.path.relpath(path=best_checkpoint, start=self.work_dir), soft_link]))
            except:
                best_checkpoint = None

            # cleanup checkpoints using the helper method
            self._clean_checkpoints(
                latest_checkpoint=os.path.join(epoch_root, "checkpoint.pt"),
                best_checkpoint=best_checkpoint,
            )

        # Start after-val operations in a separate thread
        self.after_val_thread = threading.Thread(target=after_val_ops)
        self.after_val_thread.start()

    def _find_best_checkpoint_(self) -> str:
        raise NotImplementedError("Don't know how to compare checkpoints for MultiValDatasetTrainer yet.")
