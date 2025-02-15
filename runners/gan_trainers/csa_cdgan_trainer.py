from typing import Dict
import time
import torch
from runners.gan_trainers import GAN_BaseTrainer


class CSA_CDGAN_Trainer(GAN_BaseTrainer):

    def _train_step(self, dp: Dict[str, Dict[str, torch.Tensor]]) -> None:
        # init time
        start_time = time.time()
        # do computation
        real_label = torch.ones (size=(x1.shape[0],), dtype=torch.float32, device=device)
        fake_label = torch.zeros(size=(x1.shape[0],), dtype=torch.float32, device=device)
        gen_image = self.model.generator(x)
        pred_real = self.model.discriminator(gt)
        pred_fake = self.model.discriminator(gen_image).detach()

        # compute loss
        err_d_fake = l_bce(pred_fake, fake_label)
        err_g = l_con(gen_image, gt)
        err_g_total = ct.G_WEIGHT*err_g + ct.D_WEIGHT*err_d_fake

        pred_fake_ = self.model.discriminator(gen_image.detach())
        err_d_real = l_bce(pred_real, real_label)
        err_d_fake_ = l_bce(pred_fake_, fake_label)
        err_d_total = (err_d_real + err_d_fake_) * 0.5

        # update generator
        self.optimizer.optimizers['generator'].zero_grad()
        err_g_total.backward(retain_graph = True)
        self.optimizer.optimizers['generator'].step()

        # update discriminator
        self.optimizer.optimizers['discriminator'].zero_grad()
        err_d_total.backward()
        self.optimizer.optimizers['discriminator'].step()
