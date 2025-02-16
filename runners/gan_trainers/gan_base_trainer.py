from typing import Dict, Any
from runners import BaseTrainer
from utils.builders import build_from_config


class GAN_BaseTrainer(BaseTrainer):

    def _init_optimizer_(self) -> None:
        r"""Requires self.model and self.logger.
        """
        # check dependencies
        for name in ['model', 'logger']:
            assert hasattr(self, name) and getattr(self, name) is not None
        self.logger.info("Initializing optimizer...")
        # initialize optimizer
        assert 'optimizer' in self.config, f"{self.config.keys()=}"
        optimizer_config = self.config['optimizer']
        for name in optimizer_config['args']['optimizer_cfgs']:
            params = list(getattr(self.model, name).parameters())
            optimizer_config['args']['optimizer_cfgs'][name]['args']['optimizer_config']['args']['params'] = params
        self.optimizer = build_from_config(optimizer_config)

    def _init_scheduler_(self) -> None:
        # check dependencies
        for name in ['optimizer', 'logger']:
            assert hasattr(self, name) and getattr(self, name) is not None
        self.logger.info("Initializing scheduler...")
        # initialize scheduler
        assert 'scheduler' in self.config, f"{self.config.keys()=}"
        scheduler_config = self.config['scheduler']
        for name in scheduler_config['args']['scheduler_cfgs']:
            optimizer = self.optimizer.optimizers[name].optimizer
            scheduler_config['args']['scheduler_cfgs'][name]['args']['optimizer'] = optimizer
        self.scheduler = build_from_config(scheduler_config)

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        raise NotImplementedError("GANTrainer._set_gradients_ is unused and should not be called.")
