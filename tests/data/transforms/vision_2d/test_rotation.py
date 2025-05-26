import pytest
from data.transforms.vision_2d.rotation import Rotation
import torch


@pytest.mark.parametrize(
    "input_tensor, degrees, expected_output",
    [
        # Test 90-degree counterclockwise rotation
        (
            torch.tensor([
                [1, 2],
                [3, 4]
            ]).unsqueeze(0),  # (1, 2, 2)
            90,
            torch.tensor([
                [2, 4],
                [1, 3]
            ]).unsqueeze(0),  # (1, 2, 2)
        ),
        # Test -90-degree clockwise rotation
        (
            torch.tensor([
                [1, 2],
                [3, 4]
            ]).unsqueeze(0),
            -90,
            torch.tensor([
                [3, 1],
                [4, 2]
            ]).unsqueeze(0),
        ),
        # Test 180-degree counterclockwise rotation
        (
            torch.tensor([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),  # (3, 3)
            180,
            torch.tensor([
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]),  # (3, 3)
        ),
        # Test -180-degree (which is same as 180-degree)
        (
            torch.tensor([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),
            -180,
            torch.tensor([
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]),
        ),
    ],
)
def test_rotation(input_tensor, degrees, expected_output):
    """Test Rotation transform with fixed degrees."""
    transform = Rotation(angle=degrees)
    rotated_tensor = transform(input_tensor)

    assert rotated_tensor.shape == expected_output.shape, f"Expected shape {expected_output.shape}, got {rotated_tensor.shape}"
    assert torch.equal(rotated_tensor, expected_output), f"Expected output:\n{expected_output}\nGot:\n{rotated_tensor}"
