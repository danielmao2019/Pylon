# Utils Builders — code implementation structure

## Code implementation structure trees

`utils/builders/builder.py`

```text
builder.py
├── import tempfile
├── from copy import deepcopy
├── from typing import Any
├── import easydict as edict
├── import torch
├── from torch.utils.data import Dataset
├── def build_from_config(config: Any, recursive: bool = True, **kwargs) -> Any
│   ├── # Builds the object a {class, args} config dict names, merging kwargs into its args; recursive builds the nested configs inside the args, while recursive=False instantiates over the args as-is, leaving every nested config dict unbuilt.
│   ├── if config is an edict.EasyDict
│   │   └── return config
│   ├── if config is a dict whose keys are exactly class and args
│   │   ├── calls semideepcopy  # copies the config so the merge below leaves the caller's own dict intact, preserving shared runtime objects
│   │   ├── assert kwargs is a dict
│   │   ├── assert the copy's own keys and the kwargs keys are disjoint
│   │   ├── impls update the copy's args with kwargs, a kwarg overriding a same-named args entry
│   │   ├── if recursive
│   │   │   └── for each key of the copy's args
│   │   │       └── calls build_from_config  # a nested config builds bottom-up; an already-built object comes back as itself
│   │   └── return  # the copy's class instantiated over its built args
│   ├── elif config is a dict
│   │   ├── for each key-value pair of config
│   │   │   └── calls build_from_config  # on the value; the key is carried over as-is
│   │   └── return  # a dict comprehension over the built values
│   ├── elif config is a list
│   │   ├── for each item of config
│   │   │   └── calls build_from_config
│   │   └── return  # a list comprehension over the built items
│   ├── elif config is a tuple
│   │   ├── for each item of config
│   │   │   └── calls build_from_config
│   │   └── return  # a tuple over the built items
│   └── else
│       └── return config
└── def semideepcopy(obj: Any) -> Any
    ├── # Deepcopy variant that carries shared runtime objects through as references.
    ├── if obj is one of torch.nn.Module, torch.nn.Parameter, Dataset, edict.EasyDict, tempfile.TemporaryDirectory
    │   └── return obj
    ├── elif obj is a dict
    │   ├── for each key-value pair of obj
    │   │   └── calls semideepcopy  # on the value; the key is carried over as-is
    │   └── return  # a dict comprehension over the copied values
    ├── elif obj is a list
    │   ├── for each item of obj
    │   │   └── calls semideepcopy
    │   └── return  # a list comprehension over the copied items
    ├── elif obj is a tuple
    │   ├── for each item of obj
    │   │   └── calls semideepcopy
    │   └── return  # a tuple over the copied items
    └── else
        ├── calls deepcopy
        └── return  # the deep copy
```
