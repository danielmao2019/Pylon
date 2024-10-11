import torch
from models import MultiTaskBaseModel
from models.backbones.segnet.segnet import SegNet
from models.heads.two_conv_decoder import TwoConvDecoder


class CityScapes_SegNet(MultiTaskBaseModel):
    __doc__ = r"""Used in:
    * Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
    * Conflict-Averse Gradient Descent for Multi-task Learning (https://arxiv.org/pdf/2110.14048.pdf)
    * Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
    * Multi-Task Learning as a Bargaining Game (https://arxiv.org/pdf/2202.01017.pdf)
    * Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
    """

    def __init__(self, wide: bool = False, deep: bool = False):
        backbone = SegNet(wide=wide, deep=deep)
        decoders = torch.nn.ModuleDict()
        decoders['depth_estimation'] = TwoConvDecoder(out_channels=1)
        decoders['semantic_segmentation'] = TwoConvDecoder(out_channels=19)
        decoders['instance_segmentation'] = TwoConvDecoder(out_channels=2)
        super().__init__(backbone=backbone, decoders=decoders)
