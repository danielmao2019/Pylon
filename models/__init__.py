"""
MODELS API.
"""
from models import backbones
from models.multi_task_model import MultiTaskBaseModel
from models.celeb_a_famo import CelebA_FAMO
from models.city_scapes_pspnet import CityScapes_PSPNet
from models.city_scapes_segnet import CityScapes_SegNet
from models.nyud_mt_pspnet import NYUD_MT_PSPNet
from models.nyud_mt_segnet import NYUD_MT_SegNet


__all__ = (
    'backbones',
    'MultiTaskBaseModel',
    'CelebA_FAMO',
    'CityScapes_PSPNet',
    'CityScapes_SegNet',
    'NYUD_MT_PSPNet',
    'NYUD_MT_SegNet',
)
