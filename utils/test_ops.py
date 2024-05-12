from typing import List, Dict, Union
import pytest
import numpy
import torch


# ====================================================================================================
from .ops import transpose_buffer
# ====================================================================================================

@pytest.mark.parametrize("buffer, expected", [
    (
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 2, 'c': 1}],
        {'a': [1, 3], 'b': [2, 2], 'c': [3, 1]},
    ),
])
def test_transpose_buffer(buffer: List[Dict[str, float]], expected: Dict[str, List[float]]) -> None:
    assert transpose_buffer(buffer=buffer) == expected


# ====================================================================================================
from .ops import average_buffer
# ====================================================================================================

@pytest.mark.parametrize("buffer, expected", [
    # test numpy.ndarray type
    (
        [
            {'a': numpy.array([1, 1], dtype=numpy.int64), 'b': numpy.array([2, 2], dtype=numpy.int64), 'c': numpy.array([3, 3], dtype=numpy.int64)},
            {'a': numpy.array([3, 3], dtype=numpy.int64), 'b': numpy.array([2, 2], dtype=numpy.int64), 'c': numpy.array([1, 1], dtype=numpy.int64)},
        ],
        {'a': numpy.array([2, 2], dtype=numpy.float32), 'b': numpy.array([2, 2], dtype=numpy.float32), 'c': numpy.array([2, 2], dtype=numpy.float32)},
    ),
    (
        [
            {'a': numpy.array([1, 1], dtype=numpy.float32), 'b': numpy.array([2, 2], dtype=numpy.float32), 'c': numpy.array([3, 3], dtype=numpy.float32)},
            {'a': numpy.array([3, 3], dtype=numpy.float32), 'b': numpy.array([2, 2], dtype=numpy.float32), 'c': numpy.array([1, 1], dtype=numpy.float32)},
        ],
        {'a': numpy.array([2, 2], dtype=numpy.float32), 'b': numpy.array([2, 2], dtype=numpy.float32), 'c': numpy.array([2, 2], dtype=numpy.float32)},
    ),
    # test torch.Tensor type
    (
        [
            {'a': torch.tensor([1, 1], dtype=torch.int64), 'b': torch.tensor([2, 2], dtype=torch.int64), 'c': torch.tensor([3, 3], dtype=torch.int64)},
            {'a': torch.tensor([3, 3], dtype=torch.int64), 'b': torch.tensor([2, 2], dtype=torch.int64), 'c': torch.tensor([1, 1], dtype=torch.int64)},
        ],
        {'a': torch.tensor([2, 2], dtype=torch.float32), 'b': torch.tensor([2, 2], dtype=torch.float32), 'c': torch.tensor([2, 2], dtype=torch.float32)},
    ),
    (
        [
            {'a': torch.tensor([1, 1], dtype=torch.float32), 'b': torch.tensor([2, 2], dtype=torch.float32), 'c': torch.tensor([3, 3], dtype=torch.float32)},
            {'a': torch.tensor([3, 3], dtype=torch.float32), 'b': torch.tensor([2, 2], dtype=torch.float32), 'c': torch.tensor([1, 1], dtype=torch.float32)},
        ],
        {'a': torch.tensor([2, 2], dtype=torch.float32), 'b': torch.tensor([2, 2], dtype=torch.float32), 'c': torch.tensor([2, 2], dtype=torch.float32)},
    ),
    # test list type
    (
        [
            {'a': [[1, 1], [1, 1]], 'b': [[2, 2], [2, 2]], 'c': [[3, 3], [3, 3]]},
            {'a': [[3, 3], [3, 3]], 'b': [[2, 2], [2, 2]], 'c': [[1, 1], [1, 1]]},
        ],
        {'a': [[2.0, 2.0], [2.0, 2.0]], 'b': [[2.0, 2.0], [2.0, 2.0]], 'c': [[2.0, 2.0], [2.0, 2.0]]},
    ),
    # test dict type
    (
        [
            {'a': {'aa': 1, 'ab': 2, 'ac': 3}, 'b': {'ba': 1, 'bb': 2, 'bc': 3}, 'c': {'ca': 1, 'cb': 2, 'cc': 3}},
            {'a': {'aa': 3, 'ab': 2, 'ac': 1}, 'b': {'ba': 3, 'bb': 2, 'bc': 1}, 'c': {'ca': 3, 'cb': 2, 'cc': 1}},
        ],
        {'a': {'aa': 2, 'ab': 2, 'ac': 2}, 'b': {'ba': 2, 'bb': 2, 'bc': 2}, 'c': {'ca': 2, 'cb': 2, 'cc': 2}},
    ),
    # test int type
    (
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 2, 'c': 1}],
        {'a': 2.0, 'b': 2.0, 'c': 2.0},
    ),
    # test float type
    (
        [{'a': 1.1, 'b': 2.1, 'c': 3.1}, {'a': 3.1, 'b': 2.1, 'c': 1.1}],
        {'a': 2.1, 'b': 2.1, 'c': 2.1},
    ),
    # test mixed types
    (
        [
            {'a': 1, 'b': 1.1, 'c': [1, 2], 'd': [[1, 2], [3, 4]], 'e': {'ea': 1, 'eb': 2, 'ec': 3}, 'f': numpy.array([1, 2]), 'g': torch.tensor([1, 2])},
            {'a': 1, 'b': 2.1, 'c': [2, 1], 'd': [[4, 3], [2, 1]], 'e': {'ea': 3, 'eb': 2, 'ec': 1}, 'f': numpy.array([2, 1]), 'g': torch.tensor([2, 1])},
        ],
        {'a': 1.0, 'b': 1.6, 'c': [1.5, 1.5], 'd': [[2.5, 2.5], [2.5, 2.5]], 'e': {'ea': 2, 'eb': 2, 'ec': 2}, 'f': numpy.array([1.5, 1.5]), 'g': torch.tensor([1.5, 1.5])},
    ),
])
def test_average_buffer(
    buffer: List[Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]]],
    expected: Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]],
) -> None:
    produced = average_buffer(buffer=buffer)
    assert produced.keys() == expected.keys()
    for key in produced.keys():
        assert type(produced[key]) == type(expected[key])
        if type(produced[key]) == numpy.ndarray:
            assert numpy.array_equal(produced[key], expected[key])
        elif type(produced[key]) == torch.Tensor:
            assert torch.equal(produced[key], expected[key])
        else:
            assert produced[key] == expected[key]
