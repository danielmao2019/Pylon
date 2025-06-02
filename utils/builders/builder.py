from typing import Any
from copy import deepcopy
import torch


def deepcopy_without_params(obj: Any) -> Any:
    """A version of deepcopy that preserves PyTorch parameters.
    
    Args:
        obj: The object to copy
        
    Returns:
        A deep copy of the object, but with PyTorch parameters preserved as references
    """
    if isinstance(obj, torch.nn.Parameter):
        return obj
    elif isinstance(obj, list):
        return [deepcopy_without_params(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deepcopy_without_params(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(deepcopy_without_params(item) for item in obj)
    else:
        return deepcopy(obj)


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.

    Args:
        config (Any): A config dict for building objects or any built object.
        kwargs: keyword arguments only used for building objects from `config`.
    """
    if type(config) == dict and set(config.keys()) == {'class', 'args'}:
        # Create a deep copy to avoid modifying input, but preserve parameters
        config_copy = deepcopy_without_params(config)
        
        # merge args
        assert type(kwargs) == dict, f"{type(kwargs)=}"
        assert set(config_copy.keys()) & set(kwargs.keys()) == set(), f"{config_copy.keys()=}, {kwargs.keys()=}"
        config_copy['args'].update(kwargs)
        
        # build args
        for key in config_copy['args']:
            config_copy['args'][key] = build_from_config(config_copy['args'][key])
            
        # build self
        return config_copy['class'](**config_copy['args'])
    else:
        return config
