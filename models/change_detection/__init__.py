"""
MODELS.CHANGE_DETECTION API
"""
from models.change_detection.fully_convolutional_siamese_networks import FullyConvolutionalSiameseNetwork
from models.change_detection.change_former.models.change_former_v1 import ChangeFormerV1
from models.change_detection.change_former.models.change_former_v2 import ChangeFormerV2
from models.change_detection.change_former.models.change_former_v3 import ChangeFormerV3
from models.change_detection.change_former.models.change_former_v4 import ChangeFormerV4
from models.change_detection.change_former.models.change_former_v5 import ChangeFormerV5
from models.change_detection.change_former.models.change_former_v6 import ChangeFormerV6
from models.change_detection.change_star.change_star import ChangeStar
from models.change_detection.i3pe.i3pe_model import I3PEModel
from models.change_detection.ppsl_model import PPSLModel
from models.change_detection.snunet.snunet import SNUNet
# from models.change_detection.cyws_3d.cyws_3d import CYWS3D


__all__ = (
    'FullyConvolutionalSiameseNetwork',
    'ChangeFormerV1',
    'ChangeFormerV2',
    'ChangeFormerV3',
    'ChangeFormerV4',
    'ChangeFormerV5',
    'ChangeFormerV6',
    'ChangeStar',
    'I3PEModel',
    'PPSLModel',
    # 'CYWS3D',
    'SNUNet'
)
