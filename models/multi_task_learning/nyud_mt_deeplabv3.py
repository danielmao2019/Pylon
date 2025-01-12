import torch
from models.multi_task_learning import MultiTaskBaseModel


class NYUD_MT_DeepLabV3(MultiTaskBaseModel):
    __doc__ = r"""Used in:
    * Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
    """

    def __init__(self):
        backbone = None
        decoders = torch.nn.ModuleDict()
        super().__init__(backbone=backbone, decoders=decoders)
