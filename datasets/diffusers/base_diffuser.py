"""Reference: https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/detector.py
"""
from typing import Tuple, List, Dict, Any
import math
import torch
from utils.builder import build_from_config


class BaseDiffuser(torch.utils.data.Dataset, torch.nn.Module):

    __doc__ = r"""This class defines a wrapper class on regular datasets for denoising training.
    Serves as an API bridge between datasets, models, and trainers.
    """

    def __init__(
        self,
        dataset: dict,
        num_steps: int,
        keys: List[Tuple[str, str]],
    ):
        super(BaseDiffuser, self).__init__()
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

    def _init_noise_schedule_(self, s: float = 8e-3) -> None:
        r"""cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        t = torch.linspace(start=0, end=self.num_steps, steps=self.num_steps + 1, dtype=torch.float32)
        f = torch.cos(((t / self.num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        alphas = 1 - betas
        assert torch.allclose(torch.cumprod(alphas, dim=0), alphas_cumprod[1:]), f"{(torch.cumprod(alphas, dim=0)-alphas_cumprod[1:]).abs().max()=}"
        alphas_cumprod = alphas_cumprod[1:]
        assert betas.shape == alphas.shape == alphas_cumprod.shape
        # clamp
        betas = torch.clamp(betas, min=0, max=0.999)
        alphas = torch.clamp(alphas, min=0, max=0.999)
        alphas_cumprod = torch.clamp(alphas_cumprod, min=0, max=0.999)
        # register buffers
        self.register_buffer(name='alphas', tensor=alphas)
        self.register_buffer(name='alphas_cumprod', tensor=alphas_cumprod)
        self.register_buffer(name='one_minus_alphas_cumprod', tensor=1-alphas_cumprod)

    def __len__(self) -> int:
        return len(self.dataset)

    def forward_diffusion(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert type(input) == torch.Tensor, f"{type(input)=}"
        time = torch.randint(low=0, high=self.num_steps, size=(), dtype=torch.int64)
        noise = torch.randn(size=input.shape, dtype=torch.float32, device=input.device)
        diffused = self.alphas_cumprod[time].sqrt() * input + self.one_minus_alphas_cumprod[time].sqrt() * noise
        return diffused, time

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        example = self.dataset[idx]
        for (key1, key2) in self.keys:
            noisy, time = self.forward_diffusion(example[key1][key2])
            example['inputs']['diffused_'+key2] = noisy
            example['inputs']['time'] = time
            example['labels']['original_'+key2] = example[key1].pop(key2)
            for key in example['inputs']:
                if key2 in key:
                    assert key == f"diffused_{key2}"
            for key in example['labels']:
                if key2 in key:
                    assert key == f"original_{key2}"
        return example
