from typing import Tuple, List, Dict, Callable, Union, Any


class BaseTransform:

    def __init__(self, transforms: List[Tuple[
        Callable, Union[Tuple[str, str], List[Tuple[str, str]]],
    ]]) -> None:
        r"""
        Args:
            transforms (list): the sequence of transforms to be applied onto each data point.
        """
        if transforms is None:
            transforms = []
        # input checks
        assert type(transforms) == list, f"{type(transforms)=}"
        for transform in transforms:
            assert type(transform) == tuple, f"{type(transform)=}"
            assert len(transform) == 2, f"{len(transform)=}"
            assert callable(transform[0]), f"{type(transform[0])=}"
            inputs = transform[1]
            if type(inputs) == tuple:
                assert len(inputs) == 2, f"{len(inputs)=}"
                assert type(inputs[0]) == type(inputs[1]) == str, f"{type(inputs[0])=}, {type(inputs[1])=}"
            else:
                assert type(inputs) == list, f"{type(inputs)=}"
                for keys in inputs:
                    assert type(keys) == tuple, f"{type(keys)=}"
                    assert len(keys) == 2, f"{len(keys)=}"
                    assert type(keys[0]) == type(keys[1]) == str, f"{type(keys[0])=}, {type(keys[1])=}"
        self.transforms = transforms

    def __call__(self, example: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # input checks
        assert type(example) == dict, f"{type(example)=}"
        assert set(example.keys()) == set(['inputs', 'labels', 'meta_info']), f"{example.keys()=}"
        # apply each component transform
        for i, transform in enumerate(self.transforms):
            func, inputs = transform
            try:
                if type(inputs) == tuple:
                    example[inputs[0]][inputs[1]] = func(example[inputs[0]][inputs[1]])
                else:
                    outputs = func(*(example[keys[0]][keys[1]] for keys in inputs))
                    for j, keys in enumerate(inputs):
                        example[keys[0]][keys[1]] = outputs[j]
            except Exception as e:
                raise RuntimeError(f"[ERROR] Attempting to apply self.transforms[{i}] on {inputs}: {e}")
        return example
