from typing import Set, Optional
import torch
from models import MultiTaskBaseModel
from models.heads.ppm_decoder import PyramidPoolingModule


class CityScapes_PSPNet(MultiTaskBaseModel):
    __doc__ = r"""Used in:
    * Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
    * Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
    * Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
    """

    def __init__(
        self,
        in_channels: int,
        tasks: Set[str],
        num_classes: int,
        **kwargs,
    ) -> None:
        r"""
        Args:
            num_classes (int): The number of output channels for depth estimation and instance segmentation
                are both fixed. However, that for semantic segmentation is not, depending on the experiment
                setup. The number of output channels for the semantic segmentation branch will be set
                to num_classes.
        """
        # input checks
        assert type(tasks) == set, f"{type(tasks)=}"
        assert all([type(t) == str for t in tasks])
        # initialize decoders
        decoders = torch.nn.ModuleDict()
        if "depth_estimation" in tasks:
            decoders["depth_estimation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=1)
        if "semantic_segmentation" in tasks:
            decoders["semantic_segmentation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=num_classes)
        if "instance_segmentation" in tasks:
            decoders["instance_segmentation"] = PyramidPoolingModule(in_channels=in_channels, num_classes=2)
        super(CityScapes_PSPNet, self).__init__(
            decoders=decoders, attn_in=in_channels, **kwargs,
        )
