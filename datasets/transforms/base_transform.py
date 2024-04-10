from typing import Dict, Callable, Any
import torch


class BaseTransform:

    def __init__(self, transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]]):
        if transforms is None:
            transforms = {}
        assert type(transforms) == dict, f"{type(transforms)=}"
        for key, val in transforms.items():
            assert type(key) == str, f"{type(key)=}"
            assert callable(val), f"{type(val)=}"
        self.transforms = transforms

    def __call__(self, example: Dict[str, Dict[str, Any]]):
        assert type(example) == dict, f"{type(example)=}"
        assert set(example.keys()) == set(['inputs', 'labels', 'meta_info']), f"{example.keys()=}"
        for key1 in example:
            assert type(example[key1]) == dict, f"{key1=}, {type(example[key1])=}"
            for key2 in example[key1]:
                if key2 in self.transforms:
                    try:
                        example[key1][key2] = self.transforms[key2](example[key1][key2])
                    except Exception as e:
                        raise RuntimeError(f"[ERROR] Apply transforms['{key2}'] on example['{key1}']['{key2}']: {e}")
        return example
