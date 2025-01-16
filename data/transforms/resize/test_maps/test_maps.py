import os
import pytest
import torch
from ..maps import ResizeMaps
from utils import io


@pytest.fixture
def test_image_tensor():
    """
    Fixture to load and resize a 2D image as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 2D tensor.
    """
    filepath = 'data/transforms/resize/test_maps/test_image.png'
    height, width = 1024, 1024
    if not os.path.isfile(filepath):
        pytest.skip(f"Test image not found at {filepath}.")
    return io.load_image(filepath=filepath, height=height, width=width)


@pytest.fixture
def test_tif_tensor():
    """
    Fixture to load and resize 3D .tif image bands as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 3D tensor with stacked bands.
    """
    tif_dir = 'data/transforms/resize/test_maps/test_tif/'
    if not os.path.isdir(tif_dir):
        pytest.skip(f"TIF directory not found at {tif_dir}.")
    filepaths = [os.path.join(tif_dir, filename) for filename in os.listdir(tif_dir) if filename.endswith('.tif')]
    if not filepaths:
        pytest.skip(f"No .tif files found in {tif_dir}.")
    height, width = 1024, 1024
    return io.load_image(filepaths=filepaths, height=height, width=width)


def test_2d(test_image_tensor):
    """
    Test resizing of a 2D image tensor.

    Args:
        test_image_tensor (torch.Tensor): Resized 2D tensor from the fixture.
    """
    assert test_image_tensor.ndimension() == 2, "Resized tensor should remain 2D."
    assert test_image_tensor.shape == (1024, 1024), f"Unexpected shape: {test_image_tensor.shape}"


def test_3d(test_tif_tensor):
    """
    Test resizing of a 3D image tensor.

    Args:
        test_tif_tensor (torch.Tensor): Resized 3D tensor with stacked bands from the fixture.
    """
    assert test_tif_tensor.ndimension() == 3, "Resized tensor should remain 3D."
    height, width = 1024, 1024
    assert test_tif_tensor.shape[1:] == (height, width), f"Unexpected shape: {test_tif_tensor.shape}"
