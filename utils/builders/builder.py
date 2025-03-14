from typing import Any
from copy import deepcopy


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.

    Args:
        config (Any): A config dict for building objects or any built object.
        kwargs: keyword arguments only used for building objects from `config`.
    """
    if type(config) == dict and set(config.keys()) == {'class', 'args'}:
        # Create a deep copy to avoid modifying input
        config_copy = deepcopy(config)
        
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
