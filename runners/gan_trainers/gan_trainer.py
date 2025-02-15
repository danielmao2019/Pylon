from typing import Dict, Any
import time
import torch
from runners.gan_trainers import GAN_BaseTrainer
import utils


class GANTrainer(GAN_BaseTrainer):

    def _train_step_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        # init time
        start_time = time.time()
        # do computation
        image = dp['labels']['image']
        fake_tensor = torch.zeros(
            size=(image.size(0),), dtype=torch.float32, device=image.device, requires_grad=False,
        )
        real_tensor = torch.ones(
            size=(image.size(0),), dtype=torch.float32, device=image.device, requires_grad=False,
        )
        gen_image = self.model.generator(dp['inputs'])
        # update generator
        G_loss = self.criterion(self.model.discriminator(gen_image), real_tensor)
        self.optimizer.optimizers['generator'].zero_grad()
        G_loss.backward(retain_graph=True)
        self.optimizer.optimizers['generator'].step()
        self.scheduler.schedulers['generator'].step()
        # update discriminator
        D_loss = (
            self.criterion(self.model.discriminator(image), real_tensor) +
            self.criterion(self.model.discriminator(gen_image), fake_tensor)
        ) / 2
        self.optimizer.optimizers['discriminator'].zero_grad()
        D_loss.backward(retain_graph=False)
        self.optimizer.optimizers['discriminator'].step()
        self.scheduler.schedulers['discriminator'].step()
        # update logger
        self.logger.update_buffer({"learning_rate": {
            'G': self.scheduler.schedulers['generator'].get_last_lr(),
            'D': self.scheduler.schedulers['discriminator'].get_last_lr(),
        }})
        self.logger.update_buffer(utils.logging.log_losses(losses={
            'G': G_loss, 'D': D_loss,
        }))
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})
