from typing import Dict, Any


def build_from_config(config: Dict[str, Any], **kwargs) -> Any:
    assert type(config) == dict, f"{type(config)=}"
    assert 'class' in config and 'args' in config, f"{config.keys()}"
    return config['class'](**config['args'], **kwargs)
