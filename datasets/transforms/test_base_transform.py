import pytest
from .base_transform import BaseTransform


@pytest.mark.parametrize("transforms, example, expected", [
    (
        [(lambda x: x + 1, ('inputs', 'x'))],  # transforms
        {'inputs': {'x': 0}, 'labels': {}, 'meta_info': {}},  # example
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},  # expected
    ),
    (
        [(lambda x: x + 1, ('inputs', 'x')), (lambda x: x * 2, ('inputs', 'x'))],  # transforms
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},  # example
        {'inputs': {'x': 4}, 'labels': {}, 'meta_info': {}},  # expected
    ),
    (
        [(lambda x, y: (x + 1, y + 1), [('inputs', 'a'), ('labels', 'b')])],  # transforms
        {'inputs': {'a': 0}, 'labels': {'b': 0}, 'meta_info': {}},  # example
        {'inputs': {'a': 1}, 'labels': {'b': 1}, 'meta_info': {}},  # expected
    ),
    (
        [(lambda x: x + 1, ('inputs', 'x')), (lambda x: x + 1, ('inputs', 'x'))],  # transforms
        {'inputs': {'x': 0}, 'labels': {}, 'meta_info': {}},  # example
        {'inputs': {'x': 2}, 'labels': {}, 'meta_info': {}},  # expected
    ),
])
def test_base_transform(transforms, example, expected) -> None:
    transform = BaseTransform(transforms=transforms)
    produced = transform(example)
    assert produced == expected, f"{produced=}, {expected=}"
