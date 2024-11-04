from typing import List
import pytest
from unittest.mock import patch
from .pcgrad import PCGradOptimizer
import random
import torch


@pytest.mark.parametrize("grads_list, expected", [
    (
        [
            torch.tensor([1, 1], dtype=torch.float32),
            torch.tensor([-1, 0], dtype=torch.float32),
        ],
        torch.tensor([-1/4, 3/4], dtype=torch.float32),
    ),
    (
        [
            torch.tensor([1, 1], dtype=torch.float32),
            torch.tensor([-1, 0], dtype=torch.float32),
            torch.tensor([-1, 2], dtype=torch.float32),
        ],
        torch.tensor([-1/2, 7/6], dtype=torch.float32),
    ),
])
def test_pcgrad_gradient_manipulation(grads_list: List[torch.Tensor], expected: torch.Tensor):
    prng = random.Random(0)
    pcgrad = PCGradOptimizer._pcgrad
    with patch.object(prng, 'shuffle', side_effect=lambda x: x):
        result = pcgrad(grads_list, prng)
        torch.testing.assert_close(result, expected)
