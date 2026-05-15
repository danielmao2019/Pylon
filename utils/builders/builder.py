import tempfile
from copy import deepcopy
from typing import Any

import easydict as edict
import torch
from torch.utils.data import Dataset


def semideepcopy(obj: Any) -> Any:
    """A version of deepcopy that preserves shared runtime objects.

    Args:
        obj: The object to copy.

    Returns:
        A deep copy of the object, but with shared runtime objects preserved as
        references.
    """
    if isinstance(
        obj,
        (
            torch.nn.Module,
            torch.nn.Parameter,
            Dataset,
            edict.EasyDict,
            tempfile.TemporaryDirectory,
        ),
    ):
        return obj
    elif isinstance(obj, dict):
        return {key: semideepcopy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [semideepcopy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(semideepcopy(item) for item in obj)
    else:
        return deepcopy(obj)


def build_from_config(config: Any, **kwargs) -> Any:
    """This function recursively builds objects provided by the config.

    Args:
        config (Any): A config dict for building objects or any built object.
        kwargs: keyword arguments only used for building objects from `config`.
    """
    if isinstance(config, edict.EasyDict):
        return config
    if isinstance(config, dict) and config.keys() == {'class', 'args'}:
        # Create a deep copy to avoid modifying input, but preserve parameters
        config_copy = semideepcopy(config)

        # merge args
        assert type(kwargs) == dict, f"{type(kwargs)=}"
        assert (
            set(config_copy.keys()) & set(kwargs.keys()) == set()
        ), f"{config_copy.keys()=}, {kwargs.keys()=}"
        config_copy['args'].update(kwargs)

        # build args
        for key in config_copy['args']:
            config_copy['args'][key] = build_from_config(config_copy['args'][key])

        # build self
        return config_copy['class'](**config_copy['args'])
    elif isinstance(config, dict):
        return {key: build_from_config(val) for key, val in config.items()}
    elif isinstance(config, list):
        return [build_from_config(item) for item in config]
    elif isinstance(config, tuple):
        return tuple(build_from_config(item) for item in config)
    else:
        return config
