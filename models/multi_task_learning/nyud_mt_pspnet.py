from typing import Set, Optional
import torch
from models.multi_task_learning import MultiTaskBaseModel
from models.multi_task_learning.heads.ppm_decoder import PyramidPoolingModule


class NYUD_MT_PSPNet(MultiTaskBaseModel):
    __doc__ = r"""Used in:
    * Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
    * Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_channels: int,
        tasks: Set[str],
        num_classes: Optional[int] = None,
        return_shared_rep: Optional[bool] = False,
        use_attention: Optional[bool] = False,
    ) -> None:
        # input checks
        assert type(tasks) == set, f"{type(tasks)=}"
        assert all([type(t) == str for t in tasks])
        # initialize decoders
        decoders = torch.nn.ModuleDict()
        if "depth_estimation" in tasks:
            decoders["depth_estimation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=1)
        if "normal_estimation" in tasks:
            decoders["normal_estimation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=3)
        if "semantic_segmentation" in tasks:
            decoders["semantic_segmentation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=num_classes)
        super(NYUD_MT_PSPNet, self).__init__(
            backbone=backbone, decoders=decoders, return_shared_rep=return_shared_rep,
            use_attention=use_attention, attn_in=in_channels,
        )
