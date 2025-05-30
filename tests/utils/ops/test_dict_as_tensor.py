from typing import List, Dict, Union, Any, Tuple
import pytest
import numpy
import torch


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_allclose
# ====================================================================================================

@pytest.mark.parametrize("buffer1, buffer2, expected", [
    (
        {
            'aggregated': {
                'score': 6.2039361000061035,
            },
            'per_datapoint': {
                'score': [8.609664916992188, 6.130837440490723, 5.303650379180908, 3.886174440383911, 8.193755149841309, 5.008403778076172, 2.8916900157928467, 7.274683475494385, 8.609664916992188, 6.130837440490723],
            },
        },
        {
            'aggregated': {
                'score': 6.200444221496582,
            },
            'per_datapoint': {
                'score': [8.629473686218262, 6.111517429351807, 5.312509059906006, 3.873427629470825, 8.167315483093262, 5.01690673828125, 2.894746780395508, 7.257554054260254, 8.629473686218262, 6.111517429351807],
            },
        },
        True,
    ),
])
def test_buffer_allclose(buffer1, buffer2, expected) -> None:
    assert buffer_allclose(buffer1, buffer2, rtol=1e-01, atol=0) == expected


# ====================================================================================================
from utils.ops.dict_as_tensor import transpose_buffer, buffer_permute
# ====================================================================================================

@pytest.mark.parametrize("buffer, expected", [
    # List[List[Any]] -> List[List[Any]]
    (
        [[1, 2, 3], [4, 5, 6]],
        [[1, 4], [2, 5], [3, 6]],
    ),
    # List[Dict[str, Any]] -> Dict[str, List[Any]]
    (
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 2, 'c': 1}],
        {'a': [1, 3], 'b': [2, 2], 'c': [3, 1]},
    ),
    # List[Dict[str, Dict[str, Any]]] -> Dict[str, List[Dict[str, Any]]]
    (
        [{'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}, {'a': {'x': 5, 'y': 6}, 'b': {'x': 7, 'y': 8}}],
        {'a': [{'x': 1, 'y': 2}, {'x': 5, 'y': 6}], 'b': [{'x': 3, 'y': 4}, {'x': 7, 'y': 8}]},
    ),
])
def test_transpose_buffer(buffer: List[Dict[str, Any]], expected: Dict[str, List[Any]]) -> None:
    assert transpose_buffer(buffer=buffer) == expected


@pytest.mark.parametrize("buffer", [
    [],
])
def test_transpose_buffer_invalid_cases(buffer: List[Dict[str, Any]]) -> None:
    with pytest.raises(AssertionError, match="Transpose is not supported for buffers with less than 2 axes."):
        transpose_buffer(buffer=buffer)


@pytest.mark.parametrize("buffer, axes, expected", [
    # Basic transpose (same as transpose_buffer)
    (
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],
        (1, 0),
        {'a': [1, 3], 'b': [2, 4]},
    ),
    # List[Dict[str, Any]] -> Dict[str, List[Any]]
    (
        [{'a': 1}],
        (1, 0),
        {'a': [1]},
    ),
    # List[Dict[str, List[Any]]] -> Dict[str, List[List[Any]]]
    (
        [{'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}],
        (1, 0, 2),
        {'a': [[1, 2], [5, 6]], 'b': [[3, 4], [7, 8]]},
    ),
    # List[Dict[str, List[Any]]] -> List[List[Dict[str, Any]]]
    (
        [{'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}],
        (2, 0, 1),
        [[{'a': 1, 'b': 3}, {'a': 5, 'b': 7}], [{'a': 2, 'b': 4}, {'a': 6, 'b': 8}]],
    ),
    # Tuple[Dict[str, Any], ...] -> Dict[str, Tuple[Any, ...]]
    (
        ({'a': 1, 'b': 2}, {'a': 3, 'b': 4}),
        (1, 0),
        {'a': (1, 3), 'b': (2, 4)},
    ),
    # List[Dict[str, Tuple[Any, ...]]] -> Dict[str, List[Tuple[Any, ...]]]
    (
        [{'a': (1, 2), 'b': (3, 4)}, {'a': (5, 6), 'b': (7, 8)}],
        (1, 0, 2),
        {'a': [(1, 2), (5, 6)], 'b': [(3, 4), (7, 8)]},
    ),
    # Test axes=None (reverse order)
    (
        [{'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}],
        None,
        [{'a': [1, 5], 'b': [3, 7]}, {'a': [2, 6], 'b': [4, 8]}],
    ),
])
def test_buffer_permute(buffer: Any, axes: Tuple[int, ...], expected: Any) -> None:
    assert buffer_permute(buffer, axes) == expected


@pytest.mark.parametrize("buffer, axes, expected", [
    # Empty buffer
    (
        [],
        (0,),
        [],
    ),
    # List[Dict[str, Any]] -> List[Dict[str, Any]]
    (
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],
        (0, 1),
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}],
    ),
])
def test_buffer_permute_edge_cases(buffer: Any, axes: Tuple[int, ...], expected: Any) -> None:
    assert buffer_permute(buffer, axes) == expected


@pytest.mark.parametrize("buffer, axes", [
    # Invalid axes length
    ([{'a': [1]}], (1, 0)),
    # Invalid axes values
    ([{'a': 1}], (0, 2)),
    # Duplicate axes
    ([{'a': 1}], (0, 0)),
])
def test_buffer_permute_invalid_axes(buffer: Any, axes: Tuple[int, ...]) -> None:
    with pytest.raises(AssertionError):
        buffer_permute(buffer, axes)


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_add
# ====================================================================================================

@pytest.mark.parametrize("buffers, expected", [
    (
        [{'a': {'a': 1}, 'b': {'b': 1}, 'c': {'c': 1}}, {'a': {'a': 2}, 'b': {'b': 2}, 'c': {'c': 2}}],
        {'a': {'a': 3}, 'b': {'b': 3}, 'c': {'c': 3}},
    ),
])
def test_buffer_add(buffers, expected) -> None:
    assert buffer_add(*buffers) == expected


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_sub
# ====================================================================================================

@pytest.mark.parametrize("buffer, other, expected", [
    (
        {'a': {'a': 1}, 'b': {'b': 1}, 'c': {'c': 1}},
        {'a': {'a': 2}, 'b': {'b': 2}, 'c': {'c': 2}},
        {'a': {'a': -1}, 'b': {'b': -1}, 'c': {'c': -1}},
    ),
])
def test_buffer_sub(buffer, other, expected) -> None:
    assert buffer_sub(buffer, other) == expected


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_mul
# ====================================================================================================

@pytest.mark.parametrize("buffer, other, expected", [
    (
        {'a': {'a': 2}, 'b': {'b': 2}, 'c': {'c': 2}},
        {'a': {'a': 3}, 'b': {'b': 3}, 'c': {'c': 3}},
        {'a': {'a': 6}, 'b': {'b': 6}, 'c': {'c': 6}},
    ),
])
def test_buffer_mul(buffer, other, expected) -> None:
    assert buffer_mul(buffer, other) == expected


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_div
# ====================================================================================================

@pytest.mark.parametrize("buffer, other, expected", [
    (
        {'a': {'a': 1}, 'b': {'b': 1}, 'c': {'c': 1}},
        {'a': {'a': 2}, 'b': {'b': 2}, 'c': {'c': 2}},
        {'a': {'a': 0.5}, 'b': {'b': 0.5}, 'c': {'c': 0.5}},
    ),
])
def test_buffer_div(buffer, other, expected) -> None:
    assert buffer_div(buffer, other) == expected


# ====================================================================================================
from utils.ops.dict_as_tensor import buffer_mean
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
def test_buffer_mean(
    buffer: List[Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]]],
    expected: Dict[str, Union[int, float, numpy.ndarray, torch.Tensor]],
) -> None:
    produced = buffer_mean(buffer=buffer)
    assert produced.keys() == expected.keys()
    for key in produced.keys():
        assert type(produced[key]) == type(expected[key])
        if type(produced[key]) == numpy.ndarray:
            assert numpy.array_equal(produced[key], expected[key])
        elif type(produced[key]) == torch.Tensor:
            assert torch.equal(produced[key], expected[key])
        else:
            assert produced[key] == expected[key]
