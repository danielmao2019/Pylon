from typing import Dict, Any
from runners import BaseTrainer
import time
import torch
import utils


class GANTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs) -> None:
        super(GANTrainer, self).__init__(*args, **kwargs)
        # sanity checks
        assert hasattr(self, 'model')
        assert hasattr(self.model, 'generator')
        assert hasattr(self.model, 'discriminator')
        assert hasattr(self, 'optimizer')
        assert hasattr(self.optimizer, 'G_optimizer')
        assert hasattr(self.optimizer, 'D_optimizer')
        assert hasattr(self, 'scheduler')
        assert hasattr(self.scheduler, 'G_scheduler')
        assert hasattr(self.scheduler, 'D_scheduler')

    def _train_step_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        # init time
        start_time = time.time()
        # do computation
        image = dp['labels']['image']
        fake_tensor = torch.zeros_like(image, requires_grad=False)
        real_tensor = torch.ones_like(image, requires_grad=False)
        gen_image = self.model.generator(dp['inputs'])
        # update generator
        G_loss = self.criterion(self.model.discriminator(gen_image), real_tensor)
        self.optimizer.G_optimizer.zero_grad()
        G_loss.backward()
        self.optimizer.G_optimizer.step()
        # update discriminator
        D_loss = (
            self.criterion(self.model.discriminator(image), real_tensor) +
            self.criterion(self.model.discriminator(gen_image), fake_tensor)
        ) / 2
        self.optimizer.D_optimizer.zero_grad()
        D_loss.backward()
        self.optimizer.D_optimizer.step()
        # update logger
        self.logger.update_buffer({"learning_rate": {
            'G': self.scheduler.G_scheduler.get_last_lr(),
            'D': self.scheduler.D_scheduler.get_last_lr(),
        }})
        self.logger.update_buffer(utils.logging.log_losses(losses={
            'G': G_loss, 'D': D_loss,
        }))
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})
