from typing import Dict, Any
import os
import random
import numpy
import torch


def set_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: Any):
    """Set seed for all random number generators.
    
    Args:
        seed: Any hashable object to use as seed.
    """
    if not isinstance(seed, int):
        seed = hash(seed) % (2**32)  # Ensure it's a 32-bit integer
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def get_random_states() -> Dict[str, Any]:
    """Get current states of all random number generators."""
    return {
        'python': random.getstate(),
        'numpy': numpy.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def set_random_states(states: Dict[str, Any]) -> None:
    """Set states of all random number generators.
    
    Args:
        states: Dictionary containing states for each random number generator.
    """
    random.setstate(states['python'])
    numpy.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
