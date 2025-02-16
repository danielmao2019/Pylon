"""
RUNNERS.GAN_TRAINERS API
"""
from runners.gan_trainers.gan_base_trainer import GAN_BaseTrainer
from runners.gan_trainers.gan_trainer import GANTrainer
from runners.gan_trainers.csa_cdgan_trainer import CSA_CDGAN_Trainer


__all__ = (
    'GAN_BaseTrainer',
    'GANTrainer',
    'CSA_CDGAN_Trainer',
)
