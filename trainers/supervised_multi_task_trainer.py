import torch
import optimizers
from .base_trainer import BaseTrainer
from utils.ops import apply_tensor_op
from utils.builder import build_from_config

try:
    # torch 2.x
    from torch.optim.lr_scheduler import LRScheduler
except:
    # torch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class SupervisedMultiTaskTrainer(BaseTrainer):
    __doc__ = r"""Trainer class for supervised multi-task learning.
    """

    # ====================================================================================================
    # optimizer hook related
    # ====================================================================================================

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
        dummy_example = self.train_dataloader.dataset[0]
        dummy_example = apply_tensor_op(func=lambda x: torch.stack([x.cuda(), x.cuda()], dim=0), inputs=dummy_example)
        dummy_outputs = self.model(dummy_example['inputs'])
        dummy_losses = self.criterion(y_pred=dummy_outputs, y_true=dummy_example['labels'])
        self.optimizer = build_from_config(
            losses=dummy_losses, shared_rep=dummy_outputs['shared_rep'], logger=self.logger,
            config=optimizer_config,
        )

    def _init_scheduler_(self):
        self.logger.info("Initializing scheduler...")
        assert 'scheduler' in self.config
        assert hasattr(self, 'train_dataloader') and isinstance(self.train_dataloader, torch.utils.data.DataLoader)
        assert hasattr(self, 'optimizer') and isinstance(self.optimizer, optimizers.MTLOptimizer)
        self.config['scheduler']['args']['lr_lambda'] = build_from_config(
            steps=len(self.train_dataloader), config=self.config['scheduler']['args']['lr_lambda'],
        )
        self.scheduler: LRScheduler = build_from_config(
            optimizer=self.optimizer.optimizer, config=self.config['scheduler'],
        )

    def _set_gradients_(self, example: dict):
        self.optimizer.backward(losses=example['losses'], shared_rep=example['outputs']['shared_rep'])
