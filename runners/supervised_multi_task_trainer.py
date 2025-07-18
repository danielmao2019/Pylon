from typing import Dict, Any
import torch
from .base_trainer import BaseTrainer
import optimizers
from utils.ops import apply_tensor_op
from utils.builders import build_from_config

try:
    # torch 2.x
    from torch.optim.lr_scheduler import LRScheduler
except:
    # torch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class SupervisedMultiTaskTrainer(BaseTrainer):
    __doc__ = r"""Trainer class for supervised multi-task learning.
    """

    def _init_optimizer(self) -> None:
        r"""Requires self.model.
        """
        # check dependencies
        for name in ['model', 'train_dataloader', 'criterion', 'logger']:
            assert hasattr(self, name) and getattr(self, name) is not None
        self.logger.info("Initializing optimizer...")
        # input checks
        assert 'optimizer' in self.config, f"{self.config.keys()=}"
        # initialize optimizer
        optimizer_config = self.config['optimizer']
        optimizer_config['args']['optimizer_config']['args']['params'] = list(self.model.parameters())
        dummy_dp0 = self.train_dataloader.dataset[0]
        dummy_dp1 = self.train_dataloader.dataset[1]
        dummy_example = self.train_dataloader.collate_fn([dummy_dp0, dummy_dp1])
        dummy_example = apply_tensor_op(lambda x: x.cuda(), dummy_example)
        dummy_outputs = self.model(dummy_example['inputs'])
        dummy_losses = self.criterion(y_pred=dummy_outputs, y_true=dummy_example['labels'])
        self.optimizer = build_from_config(
            losses=dummy_losses, shared_rep=dummy_outputs['shared_rep'], logger=self.logger,
            config=optimizer_config,
        )

    def _init_scheduler(self):
        # check dependencies
        for name in ['logger', 'train_dataloader', 'optimizer']:
            assert hasattr(self, name) and getattr(self, name) is not None, f"{name=}"

        self.logger.info("Initializing scheduler...")
        assert 'scheduler' in self.config
        assert isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        assert isinstance(self.optimizer, optimizers.MTLOptimizer)
        self.config['scheduler']['args']['lr_lambda'] = build_from_config(
            steps=len(self.train_dataloader), config=self.config['scheduler']['args']['lr_lambda'],
        )
        self.scheduler: LRScheduler = build_from_config(
            optimizer=self.optimizer.optimizer, config=self.config['scheduler'],
        )

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        self.optimizer.backward(losses=dp['losses'], shared_rep=dp['outputs']['shared_rep'])
