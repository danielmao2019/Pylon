from typing import Any


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.
    """
    if type(config) == dict and set(config.keys()) == set(['class', 'args']):
        for key in config['args']:
            config['args'][key] = build_from_config(config['args'][key])
        return config['class'](**config['args'], **kwargs)
    else:
        return config
