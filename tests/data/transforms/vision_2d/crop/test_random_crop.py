import pytest
from unittest.mock import patch
from data.transforms.crop.random_crop import RandomCrop
import torch


@pytest.mark.parametrize(
    "input_tensor, crop_size, mock_x, mock_y, expected_output",
    [
        # 5x5 tensor, 3x3 crop, mock x=1, y=1 (so crop starts at (1,1))
        (
            torch.tensor([
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]).unsqueeze(0),  # (1, 5, 5) shape
            (3, 3),
            1, 1,  # Mock x=1, y=1 (forcing crop to start from (1,1))
            torch.tensor([
                [7, 8, 9],
                [12, 13, 14],
                [17, 18, 19],
            ]).unsqueeze(0),  # Expected (1, 3, 3) output
        ),
        
        # 4x4 tensor, 2x2 crop, mock x=2, y=1 (crop starts at (2,1))
        (
            torch.tensor([
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                [130, 140, 150, 160],
            ]).unsqueeze(0),  # (1, 4, 4) shape
            (2, 2),
            2, 1,  # Mock x=2, y=1 (forcing crop to start from (2,1))
            torch.tensor([
                [70, 80],
                [110, 120],
            ]).unsqueeze(0),  # Expected (1, 2, 2) output
        ),
    ],
)
def test_random_crop(input_tensor, crop_size, mock_x, mock_y, expected_output):
    """Test RandomCrop with a mocked random.randint to force deterministic cropping."""
    transform = RandomCrop(size=crop_size)

    with patch("random.randint", side_effect=[mock_x, mock_y]):
        cropped_tensor = transform(input_tensor)

    assert cropped_tensor.shape == expected_output.shape, f"Expected shape {expected_output.shape}, got {cropped_tensor.shape}"
    assert torch.equal(cropped_tensor, expected_output), f"Expected output:\n{expected_output}\nGot:\n{cropped_tensor}"


def test_random_crop_bounds():
    """Test if RandomCrop respects valid crop boundaries."""
    tensor = torch.randn(3, 10, 10)  # 3-channel, 10x10 image
    crop_size = (5, 5)
    transform = RandomCrop(size=crop_size)

    for _ in range(10):  # Run multiple times to test randomness
        cropped_tensor = transform(tensor)
        assert cropped_tensor.shape[-2:] == crop_size, "Crop size does not match expected output."
    

@pytest.mark.parametrize(
    "tensor_shape, crop_size",
    [
        ((3, 5, 5), (6, 6)),  # Crop size larger than tensor dimensions
        ((1, 4, 4), (5, 2)),  # Width too large
        ((2, 6, 6), (2, 7)),  # Height too large
    ],
)
def test_random_crop_invalid_size(tensor_shape, crop_size):
    """Test if RandomCrop raises an error for invalid crop sizes."""
    tensor = torch.randn(tensor_shape)
    transform = RandomCrop(size=crop_size)

    with pytest.raises(ValueError, match="Crop size .* exceeds tensor dimensions"):
        transform(tensor)
