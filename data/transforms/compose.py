from typing import Tuple, List, Dict, Callable, Union, Any, Optional
import copy
from .base_transform import BaseTransform


class Compose(BaseTransform):

    def __init__(self, transforms: Optional[List[Tuple[
        Callable, Union[Tuple[str, str], List[Tuple[str, str]]],
    ]]] = None) -> None:
        r"""
        Args:
            transforms (list): the sequence of transforms to be applied onto each data point.
        """
        # input checks
        if transforms is None:
            transforms = []
        assert type(transforms) == list, f"{type(transforms)=}"
        for idx, transform in enumerate(transforms):
            assert type(transform) == tuple, f"{idx=}, {type(transform)=}"
            assert len(transform) == 2, f"{idx=}, {len(transform)=}"
            # check func
            func = transform[0]
            assert callable(func), f"{type(func)=}"
            # check input keys
            input_keys = transform[1]
            if type(input_keys) == tuple:
                transforms[idx] = list(transforms[idx])
                transforms[idx][1] = [transforms[idx][1]]
                transforms[idx] = tuple(transforms[idx])
                input_keys = transforms[idx][1]
            assert type(input_keys) == list, f"{type(input_keys)=}"
            for key_pair in input_keys:
                assert type(key_pair) == tuple, f"{type(key_pair)=}"
                assert len(key_pair) == 2, f"{len(key_pair)=}"
                assert type(key_pair[0]) == type(key_pair[1]) == str, f"{type(key_pair[0])=}, {type(key_pair[1])=}"
        # assign to class attribute
        self.transforms = transforms

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        r"""This method overrides parent `__call__` method.
        """
        # input checks
        assert type(datapoint) == dict, f"{type(datapoint)=}"
        assert set(datapoint.keys()) == set(['inputs', 'labels', 'meta_info']), f"{datapoint.keys()=}"
        
        # Create a deep copy of the input datapoint to avoid in-place modification
        datapoint = copy.deepcopy(datapoint)
        
        # apply each component transform
        for i, transform in enumerate(self.transforms):
            func, input_keys = transform
            try:
                if len(input_keys) == 1:
                    key_pair = input_keys[0]
                    outputs = [func(datapoint[key_pair[0]][key_pair[1]])]
                else:
                    outputs = func(*(datapoint[key_pair[0]][key_pair[1]] for key_pair in input_keys))
            except Exception as e:
                raise RuntimeError(f"Attempting to apply self.transforms[{i}] on {input_keys}: {e}")
            assert type(outputs) == list, f"{type(outputs)=}"
            assert len(outputs) == len(input_keys), f"{len(outputs)=}, {len(input_keys)=}"
            for j, key_pair in enumerate(input_keys):
                datapoint[key_pair[0]][key_pair[1]] = outputs[j]
        return datapoint
