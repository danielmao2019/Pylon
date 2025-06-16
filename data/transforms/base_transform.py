from typing import Any, Dict, List, Optional, Tuple, Union
from abc import abstractmethod
import random
import numpy as np
import torch


class BaseTransform:
    """Base class for all transforms."""

    def _get_generator(self, g_type: str, seed: Optional[Any] = None) -> Union[random.Random, np.random.Generator, torch.Generator]:
        r"""Get a generator of the specified type and seed."""
        assert isinstance(g_type, str), f"{type(g_type)=}"
        assert g_type in {'random', 'numpy', 'torch'}, f"{g_type=}"

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        if not isinstance(seed, int):
            seed = hash(seed) % (2**32)  # Ensure it's a 32-bit integer

        if g_type == 'random':
            generator = random.Random()
            generator.seed(seed)
        elif g_type == 'numpy':
            generator = np.random.Generator(np.random.PCG64(seed))
        elif g_type == 'torch':
            generator = torch.Generator(device='cuda')
            generator.manual_seed(seed)
        else:
            raise NotImplementedError(f"Unsupported generator type: {g_type}")

        return generator

    @abstractmethod
    def _call_single(self, *args, generator: torch.Generator) -> Any:
        """Apply the transform to a single input."""
        raise NotImplementedError

    def _call_single_with_generator(self, *args, generator: torch.Generator) -> Any:
        """Apply the transform to a single input with a generator."""
        # Try to call _call_single with generator first
        try:
            return self._call_single(*args, generator=generator)
        except Exception as e:
            # If error is about unexpected generator argument, try without generator
            if "got an unexpected keyword argument 'generator'" in str(e):
                return self._call_single(*args)
            else:
                raise

    def __call__(self, *args, seed: Optional[Any] = None) -> Any:
        """Apply the transform to one or more inputs."""
        assert isinstance(args, tuple), f"{type(args)=}"
        assert len(args) > 0, f"{len(args)=}"
        generator = self._get_generator(g_type='torch', seed=seed)

        if len(args) == 1:
            return self._call_single_with_generator(*args, generator=generator)
        else:
            return [self._call_single_with_generator(arg, generator=generator) for arg in args]
