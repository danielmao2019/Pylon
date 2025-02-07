from typing import Dict, Any
import torch
from .base_trainer import BaseTrainer
import optimizers
import utils


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
        self.optimizer = utils.builders.build_from_config(optimizer_config)

    def _init_scheduler_(self):
        # check dependencies
        for name in ['optimizer', 'train_dataloader', 'tot_epochs']:
            assert hasattr(self, name) and getattr(self, name) is not None
        self.logger.info("Initializing scheduler...")
        # input checks
        assert 'scheduler' in self.config
        assert isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        assert isinstance(self.optimizer, optimizers.SingleTaskOptimizer)
        assert hasattr(self.optimizer, 'optimizer') and isinstance(self.optimizer.optimizer, torch.optim.Optimizer)
        # build scheduler
        self.scheduler = utils.builders.build_scheduler(trainer=self, cfg=self.config['scheduler'])

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        r"""Set gradients in single-task learning setting.
        """
        self.optimizer.zero_grad()
        assert 'losses' in dp
        losses = dp['losses']
        assert type(losses) == torch.Tensor, f"{type(losses)=}"
        assert losses.numel() == 1, f"{losses.shape=}"
        losses.backward()
