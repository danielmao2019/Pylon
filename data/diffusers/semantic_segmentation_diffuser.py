from .base_diffuser import BaseDiffuser
from typing import Tuple, List
import torch
from utils.semantic_segmentation import to_one_hot_encoding


class SemanticSegmentationDiffuser(BaseDiffuser):

    def __init__(
        self,
        dataset: dict,
        num_steps: int,
        keys: List[Tuple[str, str]],
        num_classes: int,
        ignore_index: int,
        scale: float,
    ):
        super(SemanticSegmentationDiffuser, self).__init__(dataset=dataset, num_steps=num_steps, keys=keys)
        assert type(num_classes) == int, f"{type(num_classes)=}"
        self.num_classes = num_classes
        assert type(ignore_index) == int, f"{type(ignore_index)=}"
        self.ignore_index = ignore_index
        assert type(scale) == float, f"{type(scale)=}"
        self.scale = scale

    def forward_diffusion(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward diffusion for semantic segmentation annotations.
        This method does not utilize `BaseDiffuser.forward_diffusion` because semantic segmentation is classification task in nature.

        Args:
            mask (torch.Tensor): int64 tensor of shape (H, W). semantic segmentation labels.

        Returns:
            torch.Tensor: int64 tensor of shape (C, H, W). noisy semantic segmentation mask.
            torch.Tensor: int64 tensor of shape (). time step.
        """
        # input checks
        assert type(mask) == torch.Tensor, f"{type(mask)=}"
        assert mask.dim() == 2, f"{mask.shape=}"
        assert mask.dtype == torch.int64, f"{mask.dtype=}"
        # sample time step
        time = torch.randint(low=0, high=self.num_steps, size=(), dtype=torch.int64)
        # initialize probability distribution from mask
        probs = to_one_hot_encoding(mask, num_classes=self.num_classes, ignore_index=self.ignore_index).type(torch.float32)
        # diffuse probability distribution
        alpha_cumprod = self.alphas_cumprod[time]
        probs = alpha_cumprod * probs + (1 - alpha_cumprod) / self.num_classes
        # sample from diffused probability distribution
        probs = probs.permute(1, 2, 0)
        sample = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs).sample()
        sample = sample.permute(2, 0, 1).type(torch.int64)
        # sanity check
        assert sample.shape == (self.num_classes, mask.shape[0], mask.shape[1]), f"{sample.shape=}, {mask.shape=}"
        assert sample.dtype == torch.int64, f"{sample.dtype=}, {mask.dtype=}"
        return sample, time
