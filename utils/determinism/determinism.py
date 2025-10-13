from typing import Dict, Any
import os
import random
import numpy
import torch


def set_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: Any) -> None:
    """Set seed for all random number generators.

    Args:
        seed: Any hashable object to use as seed.
    """
    from utils.determinism.hash_utils import convert_to_seed

    seed = convert_to_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def get_random_states() -> Dict[str, Any]:
    """Get current states of all random number generators."""
    return {
        'random': random.getstate(),
        'numpy': numpy.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def set_random_states(states: Dict[str, Any]) -> None:
    """Set states of all random number generators.

    Args:
        states: Dictionary containing states for each random number generator.
    """
    random.setstate(states['random'])
    numpy.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
