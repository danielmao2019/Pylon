import pytest
import torch
from data.transforms import RandomCrop


@pytest.mark.parametrize(
    "tensor_shape, crop_size",
    [
        ((3, 5, 5), (3, 3)),  # 3-channel image, cropping 3x3
        ((1, 8, 8), (5, 5)),  # Single-channel image, cropping 5x5
        ((2, 10, 10), (6, 6)),  # Multi-channel image, cropping 6x6
        ((4, 6, 6), (2, 2)),  # 4-channel image, small crop
    ],
)
def test_random_crop_shape(tensor_shape, crop_size):
    """Test if RandomCrop produces the correct shape."""
    tensor = torch.randn(tensor_shape)  # Create a random input tensor
    transform = RandomCrop(size=crop_size)
    
    cropped_tensor = transform(tensor)
    
    expected_shape = (*tensor_shape[:-2], crop_size[1], crop_size[0])  # Preserve non-spatial dimensions
    assert cropped_tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {cropped_tensor.shape}"


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
