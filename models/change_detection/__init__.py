"""
MODELS.CHANGE_DETECTION API
"""
# Bi-Temporal Models
from models.change_detection.fc_siam.fully_convolutional_siamese_networks import FullyConvolutionalSiameseNetwork
from models.change_detection.snunet.snunet import SNUNet_ECAM
from models.change_detection.dsifn.dsifn_model import DSIFN
from models.change_detection.tiny_cd.tiny_cd_model import TinyCD
from models.change_detection import changer
from models.change_detection.change_former.models.change_former_v1 import ChangeFormerV1
from models.change_detection.change_former.models.change_former_v2 import ChangeFormerV2
from models.change_detection.change_former.models.change_former_v3 import ChangeFormerV3
from models.change_detection.change_former.models.change_former_v4 import ChangeFormerV4
from models.change_detection.change_former.models.change_former_v5 import ChangeFormerV5
from models.change_detection.change_former.models.change_former_v6 import ChangeFormerV6
from models.change_detection.change_next.models.change_next_v1 import ChangeNextV1
from models.change_detection.change_next.models.change_next_v2 import ChangeNextV2
from models.change_detection.change_next.models.change_next_v3 import ChangeNextV3
from models.change_detection.ftn.model import FTN
from models.change_detection.srcnet.srcnet_model import SRCNet
from models.change_detection.bifa.bifa_model import BiFA
from models.change_detection.cdx_former.cdx_former_model import CDXFormer
from models.change_detection import csa_cdgan
from models.change_detection.change_mamba.st_mamba_bcd import STMambaBCD
from models.change_detection.bit_cd.model import BASE_Transformer
# Single-Temporal Models
from models.change_detection.change_star.change_star import ChangeStar
from models.change_detection.i3pe.i3pe_model import I3PEModel
from models.change_detection.ppsl.ppsl_model import PPSLModel
# from models.change_detection.cyws_3d.cyws_3d import CYWS3D


__all__ = (
    # Bi-Temporal Models
    'FullyConvolutionalSiameseNetwork',
    'SNUNet_ECAM',
    'DSIFN',
    'TinyCD',
    'changer',
    'ChangeFormerV1',
    'ChangeFormerV2',
    'ChangeFormerV3',
    'ChangeFormerV4',
    'ChangeFormerV5',
    'ChangeFormerV6',
    'ChangeNextV1',
    'ChangeNextV2',
    'ChangeNextV3',
    'FTN',
    'SRCNet',
    'BiFA',
    'CDXFormer',
    'csa_cdgan',
    'STMambaBCD',
    'BASE_Transformer',
    # Single-Temporal Models
    'ChangeStar',
    'I3PEModel',
    'PPSLModel',
    # 'CYWS3D',
)
