from typing import Tuple, List, Sequence, Dict, Callable, Union, Any, Optional
import copy
from data.transforms.base_transform import BaseTransform


class Compose(BaseTransform):

    def __init__(self, transforms: Optional[List[Union[
        Tuple[
            Callable,
            Union[Tuple[str, str], List[Tuple[str, str]]],
        ],
        Dict[str, Union[
            Callable,
            Union[Tuple[str, str], List[Tuple[str, str]]],
        ]],
    ]]] = None) -> None:
        r"""
        Args:
            transforms (list): the sequence of transforms to be applied onto each data point.
                Each transform can be either:
                1. A tuple of (function, input_names) for backward compatibility
                2. A dictionary with keys:
                   - "op": The transform function
                   - "input_names": List of input key tuples or a single input key tuple
                   - "output_names": Optional list of output key tuples or a single output key tuple
        """
        # input checks
        if transforms is None:
            transforms = []
        assert type(transforms) == list, f"{type(transforms)=}"

        # Normalize all transforms using the static method
        self.transforms = [self.normalize_transforms_cfg(transform) for transform in transforms]

    @staticmethod
    def normalize_transforms_cfg(transform: Union[
        Tuple[Callable, Union[Tuple[str, str], List[Tuple[str, str]]]],
        Dict[str, Union[Callable, Union[Tuple[str, str], List[Tuple[str, str]]]]]
    ]) -> Dict[str, Any]:
        """Normalize transform configuration to standard dictionary format.

        This method converts transforms from either the legacy tuple format or the new dictionary format
        into a standardized dictionary format with keys: 'op', 'input_names', 'output_names'.

        Args:
            transform: Transform in either format:
                     - Tuple format: (transform_fn, input_keys) for backward compatibility
                     - Dictionary format: {'op': transform_fn, 'input_names': input_keys, 'output_names': output_keys}

        Returns:
            Normalized transform dictionary with keys: 'op', 'input_names', 'output_names'

        Raises:
            ValueError: If transform format is invalid
            AssertionError: If validation of keys/structure fails
        """
        if isinstance(transform, tuple):
            # Handle backward compatibility case
            assert len(transform) == 2, f"Tuple format must have exactly 2 elements, got {len(transform)}"

            # parse transform
            func = transform[0]
            assert callable(func) or isinstance(func, dict), f"Transform function must be callable or dict config, got {type(func)}"
            input_names = Compose._process_names(transform[1])
            output_names = input_names

            return {
                "op": func,
                "input_names": input_names,
                "output_names": output_names,
            }
        else:
            # Handle new dictionary format
            assert isinstance(transform, dict), f"Expected dict format, got {type(transform)}"
            assert "op" in transform, f"Transform missing 'op' key, got keys: {list(transform.keys())}"
            assert "input_names" in transform, f"Transform missing 'input_names' key, got keys: {list(transform.keys())}"

            # parse transform
            func = transform["op"]
            assert callable(func) or isinstance(func, dict), f"Transform 'op' must be callable or dict config, got {type(func)}"
            input_names = Compose._process_names(transform["input_names"])
            output_names = Compose._process_names(transform.get("output_names", input_names))

            return {
                "op": func,
                "input_names": input_names,
                "output_names": output_names,
            }

    @staticmethod
    def _process_names(names: Union[Tuple[str, str], List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        if isinstance(names, tuple):
            names = [names]
        assert isinstance(names, list), f"{type(names)=}"
        assert all(isinstance(name, tuple) for name in names), f"{type(names[0])=}"
        assert all(len(name) == 2 for name in names), f"{len(names[0])=}"
        assert all(isinstance(name[0], str) for name in names), f"{type(names[0][0])=}"
        assert all(isinstance(name[1], str) for name in names), f"{type(names[0][1])=}"
        return names

    @staticmethod
    def _call_with_seed(func: Callable, op_inputs: List[Any], seed: Optional[Any] = None) -> Sequence[Any]:
        # Try to apply transform with seed first
        try:
            if len(op_inputs) == 1:
                op_outputs = [func(op_inputs[0], seed=seed)]
            else:
                op_outputs = func(*op_inputs, seed=seed)
        except Exception as e:
            # If error is about unexpected seed argument, try without seed
            if "got an unexpected keyword argument 'seed'" in str(e):
                if len(op_inputs) == 1:
                    op_outputs = [func(op_inputs[0])]
                else:
                    op_outputs = func(*op_inputs)
            else:
                raise

        # Ensure outputs is a list
        if not isinstance(op_outputs, (tuple, list)):
            op_outputs = [op_outputs]

        return op_outputs

    def __call__(self, datapoint: Dict[str, Dict[str, Any]], seed: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        r"""This method overrides parent `__call__` method.
        """
        # input checks
        assert isinstance(datapoint, dict), f"{type(datapoint)=}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}, f"{datapoint.keys()=}"

        # Create a deep copy of the input datapoint to avoid in-place modification
        datapoint = copy.deepcopy(datapoint)

        # apply each component transform
        for i, transform in enumerate(self.transforms):
            func = transform["op"]
            input_names = transform["input_names"]
            output_names = transform["output_names"]

            op_inputs = [datapoint[key_pair[0]][key_pair[1]] for key_pair in input_names]
            op_outputs = self._call_with_seed(func, op_inputs, seed)
            assert len(op_outputs) == len(output_names), \
                f"Transform {i} produced {len(op_outputs)} outputs but expected {len(output_names)}"

            for output, key_pair in zip(op_outputs, output_names):
                datapoint[key_pair[0]][key_pair[1]] = output

        return datapoint
