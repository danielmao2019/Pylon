import os
import pytest
from PIL import Image
import torch
import numpy
from ..maps import ResizeMaps

def _pil2torch(image: Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor."""
    mode = image.mode
    # convert to torch.Tensor
    if mode == 'RGB':
        image = torch.from_numpy(numpy.array(image))
        image = image.permute(2, 0, 1)
        assert image.ndim == 3 and image.shape[0] == 3, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode == 'RGBA':
        image = torch.from_numpy(numpy.array(image))
        image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        assert image.ndim == 3 and image.shape[0] == 4, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode in ['L', 'P']:
        image = torch.from_numpy(numpy.array(image))
        assert image.ndim == 2, f"{image.shape=}"
        assert image.dtype == torch.uint8, f"{image.dtype=}"
    elif mode in ['I', 'I;16']:
        image = torch.from_numpy(numpy.array(image, dtype=numpy.int32))
        assert image.ndim == 2, f"{image.shape=}"
    else:
        raise NotImplementedError(f"Conversion from PIL image to PyTorch tensor not implemented for {mode=}.")
    return image

@pytest.fixture
def test_image_filepath():
    """
    Fixture to provide the test image filepath.

    Returns:
        str: Path to the test image.
    """
    filepath = 'data/transforms/resize/test_maps/test_image.png'
    if not os.path.isfile(filepath):
        pytest.skip(f"Test image not found at {filepath}.")
    return filepath


@pytest.fixture
def test_image_tensor(test_image_filepath):
    """
    Fixture to provide a 2D test image as a PyTorch tensor.

    Args:
        test_image_filepath (str): Path to the test image.

    Returns:
        torch.Tensor: 2D PyTorch tensor of the test image.
    """
    image = Image.open(test_image_filepath)
    return _pil2torch(image)


def test_2d(test_image_tensor):
    """
    Test resizing of a 2D image tensor.

    Args:
        test_image_tensor (torch.Tensor): 2D input image tensor.
    """
    assert test_image_tensor.ndimension() == 2, "Input tensor should be 2D."
    height, width = 1024, 1024
    resize_instance = ResizeMaps(size=(height, width))
    resized_image = resize_instance._call_single_(test_image_tensor, height, width)
    assert resized_image.ndimension() == 2, "Resized tensor should still be 2D."
    assert resized_image.shape == (height, width), (
        f"Resized tensor has unexpected shape: {resized_image.shape}"
    )