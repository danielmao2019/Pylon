import pytest
from .compose import Compose


@pytest.mark.parametrize("transforms, example, expected", [
    (
        # test single-transform single-apply
        [(lambda x: x + 1, ('inputs', 'x'))],  # transforms
        {'inputs': {'x': 0}, 'labels': {}, 'meta_info': {}},  # example
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},  # expected
    ),
    (
        # test multi-transform single-apply
        [(lambda x: x + 1, ('inputs', 'x')), (lambda x: x * 2, ('inputs', 'x'))],  # transforms
        {'inputs': {'x': 1}, 'labels': {}, 'meta_info': {}},  # example
        {'inputs': {'x': 4}, 'labels': {}, 'meta_info': {}},  # expected
    ),
    (
        # test single-transform multi-apply
        [(lambda x, y: [x + 1, y + 1], [('inputs', 'a'), ('labels', 'b')])],  # transforms
        {'inputs': {'a': 0}, 'labels': {'b': 0}, 'meta_info': {}},  # example
        {'inputs': {'a': 1}, 'labels': {'b': 1}, 'meta_info': {}},  # expected
    ),
])
def test_compose(transforms, example, expected) -> None:
    transform = Compose(transforms=transforms)
    produced = transform(example)
    assert produced == expected, f"{produced=}, {expected=}"
