import pytest
from data.transforms.compose import Compose


@pytest.fixture
def basic_datapoint():
    return {
        'inputs': {'x': 0, 'y': 1},
        'labels': {'a': 2, 'b': 3},
        'meta_info': {}
    }


@pytest.mark.parametrize("transforms, expected_parsed", [
    # Test None transforms
    (
        None,
        [],
    ),
    
    # Test empty transforms
    (
        [],
        [],
    ),
    
    # Test single-input transform (tuple format)
    (
        [(lambda x: x + 1, ('inputs', 'x'))],
        [{
            "op": lambda x: x + 1,
            "input_names": [('inputs', 'x')],
            "output_names": [('inputs', 'x')],
        }],
    ),
    
    # Test multi-input transform (tuple format)
    (
        [(lambda x, y: [x + 1, y + 1], [('inputs', 'x'), ('inputs', 'y')])],
        [{
            "op": lambda x, y: [x + 1, y + 1],
            "input_names": [('inputs', 'x'), ('inputs', 'y')],
            "output_names": [('inputs', 'x'), ('inputs', 'y')],
        }],
    ),
    
    # Test dictionary format with same input/output names
    (
        [{
            "op": lambda x: x + 1,
            "input_names": ('inputs', 'x'),
        }],
        [{
            "op": lambda x: x + 1,
            "input_names": [('inputs', 'x')],
            "output_names": [('inputs', 'x')],
        }],
    ),
    
    # Test dictionary format with different input/output names
    (
        [{
            "op": lambda x: x + 1,
            "input_names": ('inputs', 'x'),
            "output_names": ('inputs', 'x_processed'),
        }],
        [{
            "op": lambda x: x + 1,
            "input_names": [('inputs', 'x')],
            "output_names": [('inputs', 'x_processed')],
        }],
    ),
    
    # Test mixed format transforms
    (
        [
            (lambda x: x + 1, ('inputs', 'x')),
            {
                "op": lambda x: x * 2,
                "input_names": ('inputs', 'x'),
                "output_names": ('inputs', 'x_doubled'),
            }
        ],
        [
            {
                "op": lambda x: x + 1,
                "input_names": [('inputs', 'x')],
                "output_names": [('inputs', 'x')],
            },
            {
                "op": lambda x: x * 2,
                "input_names": [('inputs', 'x')],
                "output_names": [('inputs', 'x_doubled')],
            }
        ],
    ),
])
def test_compose_init(transforms, expected_parsed):
    """Test that transform configurations are correctly parsed during initialization."""
    compose = Compose(transforms=transforms)
    
    # For each transform, verify the parsed configuration
    assert len(compose.transforms) == len(expected_parsed), \
        f"Expected {len(expected_parsed)} transforms, got {len(compose.transforms)}"
    
    for actual, expected in zip(compose.transforms, expected_parsed):
        # Check that the function is the same
        assert actual["op"].__code__.co_code == expected["op"].__code__.co_code, \
            "Transform function mismatch"
        
        # Check input names
        assert actual["input_names"] == expected["input_names"], \
            f"Input names mismatch: {actual['input_names']} != {expected['input_names']}"
        
        # Check output names
        assert actual["output_names"] == expected["output_names"], \
            f"Output names mismatch: {actual['output_names']} != {expected['output_names']}"


@pytest.mark.parametrize("transforms, example, expected", [
    # Basic single-input transform tests
    (
        [(lambda x: x + 1, ('inputs', 'x'))],
        {'inputs': {'x': 0}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
    ),
    (
        [(lambda x: x * 2, ('inputs', 'x'))],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 2}, 'labels': {}, 'meta_info': {}},
    ),
    
    # Multi-input transform tests
    (
        [(lambda x: x + 1, ('inputs', 'x')), (lambda x: x * 2, ('inputs', 'x'))],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 4}, 'labels': {}, 'meta_info': {}},
    ),
    
    # Multi-input transform with multiple outputs
    (
        [(lambda x, y: [x + 1, y + 1], [('inputs', 'a'), ('labels', 'b')])],
        {'inputs': {'a': 0}, 'labels': {'b': 0}, 'meta_info': {}},
        {'inputs': {'a': 1}, 'labels': {'b': 1}, 'meta_info': {}},
    ),
    
    # New dictionary format tests
    (
        [{
            "op": lambda x: x + 1,
            "input_names": ('inputs', 'x'),
            "output_names": ('inputs', 'x_processed')
        }],
        {'inputs': {'x': 0}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 0, 'x_processed': 1}, 'labels': {}, 'meta_info': {}},
    ),
    
    # Different input/output names
    (
        [{
            "op": lambda x, y: [x + y, x - y],
            "input_names": [('inputs', 'x'), ('inputs', 'y')],
            "output_names": [('labels', 'sum'), ('labels', 'diff')]
        }],
        {'inputs': {'x': 5, 'y': 3}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 5, 'y': 3}, 'labels': {'sum': 8, 'diff': 2}, 'meta_info': {}},
    ),
    
    # Mixed old and new format
    (
        [
            (lambda x: x + 1, ('inputs', 'x')),
            {
                "op": lambda x: x * 2,
                "input_names": ('inputs', 'x'),
                "output_names": ('inputs', 'x_doubled')
            }
        ],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 2, 'x_doubled': 4}, 'labels': {}, 'meta_info': {}},
    ),
    
    # Empty transforms
    (
        [],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
    ),
])
def test_compose_valid_transforms(transforms, example, expected):
    """Test valid transform configurations."""
    transform = Compose(transforms=transforms)
    produced = transform(example)
    assert produced == expected, f"{produced=}, {expected=}"


@pytest.mark.parametrize("transforms, example, error_type, error_msg", [
    # Invalid transform type
    (
        [123],  # Not a tuple or dict
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "idx=0, type(transform)=<class 'int'>",
    ),
    
    # Invalid tuple length
    (
        [(lambda x: x + 1, ('inputs', 'x'), 'extra')],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "idx=0, len(transform)=3",
    ),
    
    # Non-callable function
    (
        [("not_a_function", ('inputs', 'x'))],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "type(func)=<class 'str'>",
    ),
    
    # Invalid input names type
    (
        [(lambda x: x + 1, "not_a_tuple")],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "type(names)=<class 'str'>",
    ),
    
    # Invalid key pair length
    (
        [(lambda x: x + 1, [('inputs', 'x', 'extra')])],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "len(names[0])=3",
    ),
    
    # Invalid key type
    (
        [(lambda x: x + 1, [(123, 'x')])],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "type(names[0][0])=<class 'int'>",
    ),
    
    # Missing dictionary keys
    (
        [{"input_names": ('inputs', 'x')}],  # Missing "op"
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        AssertionError,
        "Transform 0 missing 'op' key",
    ),
    
    # Mismatched output count
    (
        [{
            "op": lambda x: [x + 1, x + 2],  # Returns 2 outputs
            "input_names": ('inputs', 'x'),
            "output_names": [('inputs', 'x1'), ('inputs', 'x2'), ('inputs', 'x3')]  # Expects 3 outputs
        }],
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},
        RuntimeError,
        "Transform 0 produced 1 outputs but expected 3",
    ),
    
    # Invalid datapoint structure
    (
        [(lambda x: x + 1, ('inputs', 'x'))],
        {'wrong_key': {'x': 1}},  # Missing required keys
        AssertionError,
        "datapoint.keys()=dict_keys(['wrong_key'])",
    ),
])
def test_compose_invalid_inputs(transforms, example, error_type, error_msg):
    """Test invalid transform configurations and error handling."""
    import re
    with pytest.raises(error_type, match=re.escape(error_msg)):
        transform = Compose(transforms=transforms)
        transform(example)


def test_compose_none_transforms():
    """Test that None transforms are handled correctly."""
    transform = Compose(transforms=None)
    example = {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}}
    expected = {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}}
    produced = transform(example)
    assert produced == expected


def test_compose_transform_error_propagation():
    """Test that transform function errors are properly propagated."""
    def failing_transform(x):
        raise ValueError("Transform failed")
    
    transform = Compose(transforms=[(failing_transform, ('inputs', 'x'))])
    example = {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}}
    
    with pytest.raises(RuntimeError, match="Transform failed"):
        transform(example)


def test_compose_deep_copy():
    """Test that input datapoint is not modified in-place."""
    example = {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}}
    original = {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}}
    
    transform = Compose(transforms=[(lambda x: x + 1, ('inputs', 'x'))])
    transform(example)
    
    assert example == original, "Input datapoint was modified in-place"
