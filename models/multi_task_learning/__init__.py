"""
MODELS.MULTI_TASK_LEARNING API
"""
from models.multi_task_learning.multi_task_model import MultiTaskBaseModel
from models.multi_task_learning.multi_mnist_lenet5 import MultiMNIST_LeNet5
from models.multi_task_learning.celeb_a_famo import CelebA_FAMO
from models.multi_task_learning.celeb_a_resnet18 import CelebA_ResNet18
from models.multi_task_learning.city_scapes_pspnet import CityScapes_PSPNet
from models.multi_task_learning.city_scapes_segnet import CityScapes_SegNet
from models.multi_task_learning.nyud_mt_pspnet import NYUD_MT_PSPNet
from models.multi_task_learning.nyud_mt_segnet import NYUD_MT_SegNet


__all__ = (
    'MultiTaskBaseModel',
    'MultiMNIST_LeNet5',
    'CelebA_FAMO',
    'CelebA_ResNet18',
    'CityScapes_PSPNet',
    'CityScapes_SegNet',
    'NYUD_MT_PSPNet',
    'NYUD_MT_SegNet',
)
