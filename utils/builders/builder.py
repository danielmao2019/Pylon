import tempfile
from copy import deepcopy
from typing import Any

import easydict as edict
import torch
from torch.utils.data import Dataset


def build_from_config(config: Any, recursive: bool = True, **kwargs) -> Any:
    """Builds the object a `{class, args}` config dict names, merging kwargs into its args.

    Args:
        config: A config dict for building objects, a container of such configs,
            or any already-built object (returned unchanged).
        recursive: Whether the args-building recursion runs: `True` builds the
            nested configs inside the args bottom-up, while `False` instantiates
            over the args as-is, leaving every nested config dict unbuilt.
        kwargs: Keyword arguments merged into the config's args before building,
            a kwarg overriding a same-named args entry.

    Returns:
        The built object, the input container rebuilt over its built items, or
        `config` itself when it is already a built leaf.
    """
    if isinstance(config, edict.EasyDict):
        return config
    if isinstance(config, dict) and config.keys() == {'class', 'args'}:
        # Copies the config so the input is never mutated, preserving shared
        # runtime objects.
        config_copy = semideepcopy(config)
        assert type(kwargs) == dict, f"{type(kwargs)=}"
        assert (
            set(config_copy.keys()) & set(kwargs.keys()) == set()
        ), f"{config_copy.keys()=}, {kwargs.keys()=}"
        config_copy['args'].update(kwargs)
        if recursive:
            # A nested config builds bottom-up; an already-built object passes
            # through unchanged.
            for key in config_copy['args']:
                config_copy['args'][key] = build_from_config(config_copy['args'][key])
        return config_copy['class'](**config_copy['args'])
    elif isinstance(config, dict):
        return {key: build_from_config(val) for key, val in config.items()}
    elif isinstance(config, list):
        return [build_from_config(item) for item in config]
    elif isinstance(config, tuple):
        return tuple(build_from_config(item) for item in config)
    else:
        return config


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
