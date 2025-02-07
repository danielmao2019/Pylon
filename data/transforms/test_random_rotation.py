import pytest
from unittest.mock import patch
from .random_rotation import RandomRotation
import torch


@pytest.mark.parametrize(
    "choices, range_values, mock_degree, input_tensor, expected_output",
    [
        # Choices-based test: Mock rotation with 90 degrees
        (
            [0, 90, 180, 270], None,
            90,
            torch.tensor([
                [1, 2],
                [3, 4]
            ]).unsqueeze(0),
            torch.tensor([
                [2, 4],
                [1, 3]
            ]).unsqueeze(0),
        ),
        # Choices-based test: Mock rotation with -90 degrees
        (
            [0, 90, -90, 180, -180], None,
            -90,
            torch.tensor([
                [1, 2],
                [3, 4]
            ]).unsqueeze(0),
            torch.tensor([
                [3, 1],
                [4, 2]
            ]).unsqueeze(0),
        ),
        # Range-based test: Mock rotation with 180 degrees
        (
            None, (90, 180), 180,
            torch.tensor([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),
            torch.tensor([
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]),
        ),
        # Range-based test: Mock rotation with -180 degrees
        (
            None, (-180, 0), -180,
            torch.tensor([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),
            torch.tensor([
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]),
        ),
    ],
)
def test_random_rotation(choices, range_values, mock_degree, input_tensor, expected_output):
    """Test RandomRotation with mocked random selection."""
    transform = RandomRotation(choices=choices, range=range_values)

    with patch("random.choice", return_value=mock_degree) if choices else patch("random.randint", return_value=mock_degree):
        rotated_tensor = transform(input_tensor)

    assert rotated_tensor.shape == expected_output.shape, f"Expected shape {expected_output.shape}, got {rotated_tensor.shape}"
    assert torch.equal(rotated_tensor, expected_output), f"Expected output:\n{expected_output}\nGot:\n{rotated_tensor}"
