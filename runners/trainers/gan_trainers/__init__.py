"""
RUNNERS.GAN_TRAINERS API
"""
from runners.trainers.gan_trainers.gan_base_trainer import GAN_BaseTrainer
from runners.trainers.gan_trainers.gan_trainer import GANTrainer
from runners.trainers.gan_trainers.csa_cdgan_trainer import CSA_CDGAN_Trainer


__all__ = (
    'GAN_BaseTrainer',
    'GANTrainer',
    'CSA_CDGAN_Trainer',
)
