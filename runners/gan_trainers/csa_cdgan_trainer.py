from typing import Dict
import torch
from runners.gan_trainers import GAN_BaseTrainer


class CSA_CDGAN_Trainer(GAN_BaseTrainer):

    def _train_step(self, dp: Dict[str, Dict[str, torch.Tensor]]) -> None:
        pass
