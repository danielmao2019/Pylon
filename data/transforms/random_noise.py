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
        # Generate random noise with the same shape as the input
        noise = torch.randn_like(tensor, generator=generator) * self.std
        # Add the noise to the input
        return tensor + noise
