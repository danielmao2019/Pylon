from typing import Dict
import time
import torch
from runners.gan_trainers import GAN_BaseTrainer
import utils


class CSA_CDGAN_Trainer(GAN_BaseTrainer):

    def _train_step_(self, dp: Dict[str, Dict[str, torch.Tensor]]) -> None:
        # init time
        start_time = time.time()

        # compute outputs
        gen_image = self.model.generator(dp['inputs'])
        change_map = dp['labels']['change_map']
        change_map_one_hot = torch.eye(gen_image.size(1), dtype=torch.float32, device=change_map.device)[change_map].permute(0, 3, 1, 2)
        dp['labels']['change_map'] = change_map_one_hot
        dp['outputs'] = {
            'gen_image': gen_image,
            'pred_real': self.model.discriminator(dp['labels']['change_map']),
            'pred_fake_g': self.model.discriminator(gen_image).detach(),
            'pred_fake_d': self.model.discriminator(gen_image.detach()),
        }

        # prepare labels
        real_label = torch.ones(
            size=(dp['inputs']['img_1'].shape[0],), dtype=torch.float32, device=self.device, requires_grad=False,
        )
        fake_label = torch.zeros(
            size=(dp['inputs']['img_1'].shape[0],), dtype=torch.float32, device=self.device, requires_grad=False,
        )
        dp['labels'].update({'real_label': real_label, 'fake_label': fake_label})

        # update generator
        self.optimizer.optimizers['generator'].zero_grad()
        g_loss = self.criterion.task_criteria['generator'](y_pred=dp['outputs'], y_true=dp['labels'])
        g_loss.backward(retain_graph=True)
        self.optimizer.optimizers['generator'].step()
        self.scheduler.schedulers['generator'].step()

        # update discriminator
        self.optimizer.optimizers['discriminator'].zero_grad()
        d_loss = self.criterion.task_criteria['discriminator'](y_pred=dp['outputs'], y_true=dp['labels'])
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
