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

    def __str__(self) -> str:
        """String representation of the transform."""
        class_name = self.__class__.__name__

        # Try to get constructor parameters if available
        if hasattr(self, '__dict__') and self.__dict__:
            # Filter out private attributes
            public_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            formatted_params = self.format_params(public_attrs)

            if formatted_params:
                return f"{class_name}({formatted_params})"

        return class_name

    @staticmethod
    def format_params(params: Dict[str, Any]) -> str:
        """Format parameters into a string representation.

        Args:
            params: Dictionary of parameters to format

        Returns:
            Formatted parameter string (e.g., "size=(96, 96), interpolation=None")
        """
        if not params:
            return ""

        formatted_params = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                formatted_params.append(f"{key}={value}")
            elif isinstance(value, str):
                formatted_params.append(f"{key}='{value}'")
            elif isinstance(value, (list, tuple)):
                if len(value) <= 3:
                    formatted_params.append(f"{key}={value}")
                else:
                    formatted_params.append(f"{key}=[...{len(value)} items]")
            elif value is None:
                formatted_params.append(f"{key}=None")
            elif hasattr(value, '__str__') and hasattr(value, '__class__') and hasattr(value.__class__, '__name__'):
                # For objects with meaningful string representations (like nested transforms)
                value_str = str(value)
                # Only use the string representation if it's more informative than just the class name
                if value_str != value.__class__.__name__ and not value_str.startswith('<'):
                    formatted_params.append(f"{key}={value_str}")
                else:
                    formatted_params.append(f"{key}={type(value).__name__}")
            else:
                formatted_params.append(f"{key}={type(value).__name__}")

        return ", ".join(formatted_params)
