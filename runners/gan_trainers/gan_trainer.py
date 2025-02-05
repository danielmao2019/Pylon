from typing import Dict, Any
from runners import BaseTrainer
import time
import torch
import utils
from utils.builder import build_from_config
import schedulers


class GANTrainer(BaseTrainer):

    def _init_optimizer_(self) -> None:
        r"""Requires self.model and self.logger.
        """
        # check dependencies
        for name in ['model', 'logger']:
            assert hasattr(self, name) and getattr(self, name) is not None
        self.logger.info("Initializing optimizer...")
        # input checks
        assert 'optimizer' in self.config, f"{self.config.keys()=}"
        # initialize optimizer
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
        # input checks
        assert 'scheduler' in self.config, f"{self.config.keys()=}"
        # build lr lambda
        self.G_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer.optimizers['generator'].optimizer,
            lr_lambda=schedulers.ConstantLambda(),
        )
        self.D_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer.optimizers['discriminator'].optimizer,
            lr_lambda=schedulers.ConstantLambda(),
        )

    def _train_step_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        # init time
        start_time = time.time()
        # do computation
        image = dp['labels']['image']
        fake_tensor = torch.zeros(size=(image.size(0),), device=image.device, requires_grad=False)
        real_tensor = torch.ones(size=(image.size(0),), device=image.device, requires_grad=False)
        gen_image = self.model.generator(dp['inputs'])
        # update generator
        G_loss = self.criterion(self.model.discriminator(gen_image), real_tensor)
        self.optimizer.optimizers['generator'].zero_grad()
        G_loss.backward(retain_graph=True)
        self.optimizer.optimizers['generator'].step()
        self.G_scheduler.step()
        # update discriminator
        D_loss = (
            self.criterion(self.model.discriminator(image), real_tensor) +
            self.criterion(self.model.discriminator(gen_image), fake_tensor)
        ) / 2
        self.optimizer.optimizers['discriminator'].zero_grad()
        D_loss.backward(retain_graph=False)
        self.optimizer.optimizers['discriminator'].step()
        self.D_scheduler.step()
        # update logger
        self.logger.update_buffer({"learning_rate": {
            'G': self.scheduler.G_scheduler.get_last_lr(),
            'D': self.scheduler.D_scheduler.get_last_lr(),
        }})
        self.logger.update_buffer(utils.logging.log_losses(losses={
            'G': G_loss, 'D': D_loss,
        }))
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        raise NotImplementedError("GANTrainer._set_gradients_ is unused and should not be called.")
