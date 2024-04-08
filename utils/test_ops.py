from typing import List, Dict
import pytest
from .ops import transpose_buffer


@pytest.mark.parametrize("input_, expected", [
    (
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 3, 'b': 2, 'c': 1}],
        {'a': [1, 3], 'b': [2, 2], 'c': [3, 1]},
    ),
])
def test_transpose_buffer(input_: List[Dict[str, float]], expected: Dict[str, List[float]]):
    assert transpose_buffer(buffer=input_) == expected
