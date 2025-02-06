import pytest
from .flip import Flip
import torch


@pytest.mark.parametrize(
    "tensor, axis, expected",
    [
        # Test flipping along the last axis (columns)
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), 1, torch.tensor([[3, 2, 1], [6, 5, 4]])),

        # Test flipping along the first axis (rows)
        (torch.tensor([[1, 2], [3, 4], [5, 6]]), 0, torch.tensor([[5, 6], [3, 4], [1, 2]])),

        # Test flipping along a 3D tensor (depth)
        (
            torch.arange(2 * 2 * 2).view(2, 2, 2), 
            2, 
            torch.tensor([[[1, 0], [3, 2]], [[5, 4], [7, 6]]])
        ),

        # Test flipping a 1D tensor
        (torch.tensor([1, 2, 3, 4]), 0, torch.tensor([4, 3, 2, 1])),

        # Test flipping along negative axis (-1 means last axis)
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), -1, torch.tensor([[3, 2, 1], [6, 5, 4]])),
    ],
)
def test_flip(tensor, axis, expected):
    flip_op = Flip(axis)
    result = flip_op(tensor)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "tensor, axis",
    [
        (torch.tensor([[1, 2], [3, 4]]), 2),  # Out of bounds axis for 2D tensor
        (torch.tensor([1, 2, 3]), -4),  # Negative axis out of bounds for 1D tensor
    ],
)
def test_flip_invalid_axis(tensor, axis):
    flip_transform = Flip(axis)
    with pytest.raises(ValueError, match="Axis .* is out of bounds"):
        flip_transform._call_single_(tensor)
