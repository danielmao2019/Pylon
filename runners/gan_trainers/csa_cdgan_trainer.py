from typing import Dict
import time
import torch
from runners.gan_trainers import GAN_BaseTrainer
import utils


class CSA_CDGAN_Trainer(GAN_BaseTrainer):

    def _train_step_(self, dp: Dict[str, Dict[str, torch.Tensor]]) -> None:
        # init time
        start_time = time.time()
        real_label = torch.ones(
            size=(dp['inputs']['img_1'].shape[0],), dtype=torch.float32, device=self.device, requires_grad=False,
        )
        fake_label = torch.zeros(
            size=(dp['inputs']['img_1'].shape[0],), dtype=torch.float32, device=self.device, requires_grad=False,
        )
        dp['labels']['change_map'] = dp['labels']['change_map'].unsqueeze(1).to(torch.float32)

        # compute outputs
        gen_image = self.model.generator(dp['inputs'])
        pred_real = self.model.discriminator(dp['labels']['change_map'])

        # prepare labels
        g_loss = torch.nn.L1Loss()(gen_image, dp['labels']['change_map'])

        # update discriminator
        pred_fake_d = self.model.discriminator(gen_image.detach())
        err_d_real = torch.nn.BCELoss()(pred_real, real_label)
        err_d_fake = torch.nn.BCELoss()(pred_fake_d, fake_label)
        d_loss = (err_d_real + err_d_fake) * 0.5

        # update generator
        self.optimizer.optimizers['generator'].zero_grad()
        g_loss.backward(retain_graph=True)
        self.optimizer.optimizers['generator'].step()
        self.scheduler.schedulers['generator'].step()

        self.optimizer.optimizers['discriminator'].zero_grad()
        d_loss.backward()
        self.optimizer.optimizers['discriminator'].step()
        self.scheduler.schedulers['discriminator'].step()

        # update logger
        self.logger.update_buffer({"learning_rate": {
            'G': self.scheduler.schedulers['generator'].get_last_lr(),
            'D': self.scheduler.schedulers['discriminator'].get_last_lr(),
        }})
        self.logger.update_buffer(utils.logging.log_losses(losses={
            'G': g_loss, 'D': d_loss,
        }))
        self.logger.update_buffer({"iteration_time": round(time.time() - start_time, 2)})
