import torch
from models import MultiTaskBaseModel


class CityScapes_DeepLabV3(MultiTaskBaseModel):
    __doc__ = r"""Used in:
    * Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (https://arxiv.org/pdf/1705.07115.pdf)
    * Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
    """

    def __init__(self):
        backbone = None
        decoders = torch.nn.ModuleDict()
        super().__init__(backbone=backbone, decoders=decoders)
