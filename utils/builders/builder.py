from typing import Any


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.

    Args:
        config (Any): A config dict for building objects or any built object.
        kwargs: keyword arguments only used for building objects from `config`.
    """
    if type(config) == dict and set(config.keys()) == set(['class', 'args']):
        # merge args
        assert type(kwargs) == dict, f"{type(kwargs)=}"
        assert set(config.keys()) & set(kwargs.keys()) == set(), f"{config.keys()=}, {kwargs.keys()=}"
        config['args'].update(kwargs)
        # build args
        for key in config['args']:
            config['args'][key] = build_from_config(config['args'][key])
        # build self
        return config['class'](**config['args'])
    else:
        return config
