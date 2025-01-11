"""
MODELS API
"""
from models import backbones
from models.multi_task_model import MultiTaskBaseModel
from models.multi_mnist_lenet5 import MultiMNIST_LeNet5
from models.celeb_a_famo import CelebA_FAMO
from models.celeb_a_resnet18 import CelebA_ResNet18
from models.city_scapes_pspnet import CityScapes_PSPNet
from models.city_scapes_segnet import CityScapes_SegNet
from models.nyud_mt_pspnet import NYUD_MT_PSPNet
from models.nyud_mt_segnet import NYUD_MT_SegNet
from models import change_detection


__all__ = (
    'backbones',
    'MultiTaskBaseModel',
    'MultiMNIST_LeNet5',
    'CelebA_FAMO',
    'CelebA_ResNet18',
    'CityScapes_PSPNet',
    'CityScapes_SegNet',
    'NYUD_MT_PSPNet',
    'NYUD_MT_SegNet',
    'change_detection',
)
