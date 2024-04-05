"""Reference: https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/detector.py
"""
from typing import Tuple, List, Dict, Any
import math
import torch
from utils.builder import build_from_config


class DiffusionDatasetWrapper(torch.utils.data.Dataset):

    __doc__ = r"""This class defines a wrapper class on regular datasets for denoising training.
    Serves as an API bridge between datasets, models, and trainers.
    """

    def __init__(
        self,
        dataset: dict,
        num_steps: int,
        keys: List[Tuple[str, str]],
    ):
        super(DiffusionDatasetWrapper, self).__init__()
        self.dataset = build_from_config(dataset)
        assert type(num_steps) == int, f"{type(num_steps)=}"
        self.num_steps = num_steps
        self._init_keys_(keys)
        self._init_noise_schedule_()

    def _init_keys_(self, keys: List[Tuple[str, str]]) -> None:
        assert type(keys) == list, f"{type(keys)=}"
        assert len(set(keys)) == len(keys), f"{keys=}"
        for key_seq in keys:
            assert type(key_seq) == tuple, f"{type(key_seq)=}"
            assert len(key_seq) == 2
            assert type(key_seq[0]) == type(key_seq[1]) == str
            assert key_seq[0] in ['inputs', 'labels']
        self.keys = keys

    def _init_noise_schedule_(self, s: float = 8e-3):
        r"""cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        x = torch.linspace(start=0, end=self.num_steps, steps=self.num_steps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / self.num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
        self.alphas = 1 - betas
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def __len__(self) -> int:
        return len(self.dataset)

    def forward_diffusion(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert type(input) == torch.Tensor, f"{type(input)=}"
        time = torch.randint(low=0, high=self.num_steps, size=(), dtype=torch.int64)
        noise = torch.randn(size=input.shape, dtype=torch.float32, device=input.device)
        diffused = self.sqrt_alphas_cumprod[time] * input + self.sqrt_one_minus_alphas_cumprod[time] * noise
        return diffused, time

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        example = self.dataset[idx]
        for (key1, key2) in self.keys:
            if key1 == 'inputs':
                label = example['inputs'][key2]
                example['labels'][key2] = label
                noisy, time = self.forward_diffusion(label)
                example['inputs'][key2] = noisy
                example['inputs']['time'] = time
            elif key1 == 'labels':
                noisy, time = self.forward_diffusion(example['labels'][key2])
                example['inputs'][key2] = noisy
                example['inputs']['time'] = time
            else:
                raise ValueError(f"[ERROR] Unrecognized key {key1}.")
        return example
