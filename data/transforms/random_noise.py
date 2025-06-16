from typing import Any
import torch
from data.transforms import BaseTransform


class RandomNoise(BaseTransform):
    """Adds random noise to tensors for testing transform randomness."""

    def __init__(self, std: float = 0.1) -> None:
        """
        Initialize random noise transform.

        Args:
            std (float): Standard deviation of the noise to add. Default: 0.1
        """
        self.std = std

    def _call_single(self, tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        assert tensor.device == generator.device, f"{tensor.device=}, {generator.device=}"
        # Generate random noise with the same shape as the input
        noise = torch.randn(size=tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator) * self.std
        # Add the noise to the input
        return tensor + noise
