from typing import Dict, Any
import torch
from .base_trainer import BaseTrainer
import optimizers
from utils.builders import build_from_config

try:
    # torch 2.x
    from torch.optim.lr_scheduler import LRScheduler
except:
    # torch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class SupervisedSingleTaskTrainer(BaseTrainer):
    __doc__ = r"""Trainer class for supervised single-task learning.
    """

    def _init_optimizer_(self) -> None:
        r"""Requires self.model.
        """
        self.logger.info("Initializing optimizer...")
        # input checks
        assert 'optimizer' in self.config, f"{self.config.keys()=}"
        # check dependencies
        for name in ['model', 'train_dataloader', 'criterion', 'logger']:
            assert hasattr(self, name) and getattr(self, name) is not None
        # initialize optimizer
        optimizer_config = self.config['optimizer']
        optimizer_config['args']['optimizer_config']['args']['params'] = self.model.parameters()
        self.optimizer = build_from_config(optimizer_config)

    def _init_scheduler_(self):
        self.logger.info("Initializing scheduler...")
        assert 'scheduler' in self.config
        # build lr lambda
        assert hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        self.config['scheduler']['args']['lr_lambda'] = build_from_config(
            steps=len(self.train_dataloader), config=self.config['scheduler']['args']['lr_lambda'],
        )
        # build scheduler
        assert hasattr(self, 'optimizer') and isinstance(self.optimizer, optimizers.SingleTaskOptimizer)
        assert hasattr(self.optimizer, 'optimizer') and isinstance(self.optimizer.optimizer, torch.optim.Optimizer)
        self.scheduler: LRScheduler = build_from_config(
            optimizer=self.optimizer.optimizer, config=self.config['scheduler'],
        )

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        r"""Set gradients in single-task learning setting.
        """
        self.optimizer.zero_grad()
        assert 'losses' in dp
        losses = dp['losses']
        assert type(losses) == torch.Tensor, f"{type(losses)=}"
        assert losses.numel() == 1, f"{losses.shape=}"
        losses.backward()
