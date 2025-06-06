from typing import Any, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch


class BaseTransform:
    """Base class for all transforms."""

    def set_seed(self, seed: Any) -> None:
        """Set the random seed for the generator if it exists."""
        if hasattr(self, 'generator'):
            if not isinstance(seed, int):
                seed = hash(seed) % (2**32)  # Ensure it's a 32-bit integer
            if isinstance(self.generator, random.Random):
                self.generator.seed(seed)
            elif isinstance(self.generator, np.random.Generator):
                self.generator = np.random.Generator(np.random.PCG64(seed))
            elif isinstance(self.generator, torch.Generator):
                self.generator.manual_seed(seed)
            else:
                raise TypeError(f"Unsupported generator type: {type(self.generator)}")

    def _call_single_(self, *args) -> Any:
        """Apply the transform to a single input."""
        raise NotImplementedError

    def __call__(self, *args) -> Any:
        """Apply the transform to one or more inputs."""
        assert isinstance(args, tuple), f"{type(args)=}"
        assert len(args) > 0, f"{len(args)=}"
        if len(args) == 1:
            return self._call_single_(*args)
        else:
            return [self._call_single_(arg) for arg in args]
