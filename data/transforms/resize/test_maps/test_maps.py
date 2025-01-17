from typing import List
import os
import glob
import pytest
import torch
from ..maps import ResizeMaps
from PIL import Image
from utils.io import _pil2torch, _load_bands


@pytest.fixture
def test_image_2d():
    """
    Fixture to load and resize a 2D image as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 2D tensor.
    """
    filepath = "./data/transforms/resize/test_maps/assets/test_png.png"
    if not os.path.isfile(filepath):
        pytest.skip(f"Test image not found at {filepath}.")
    image = _pil2torch(Image.open(filepath))
    assert image.shape == (1024, 1024), f"{image.shape=}"
    return image


@pytest.fixture
def test_image_3d():
    """
    Fixture to load and resize 3D .tif image bands as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 3D tensor with stacked bands.
    """
    tif_dir = 'data/transforms/resize/test_maps/assets'
    if not os.path.isdir(tif_dir):
        pytest.skip(f"TIF directory not found at {tif_dir}.")
    filepaths = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
    if not filepaths:
        pytest.skip(f"No .tif files found in {tif_dir}.")
    image = _load_bands(filepaths=filepaths, height=512, width=512)
    assert image.shape == (2, 512, 512)
    return image


def test_resize_maps_2d(test_image_2d):
    """
    Test resizing of a 2D image tensor.

    Args:
        test_image_tensor (torch.Tensor): Resized 2D tensor from the fixture.
    """
    new_height, new_width = 256, 256
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_2d)
    assert resized_image.shape == (new_height, new_width), f"{resized_image.shape=}"


def test_3d(test_image_3d):
    """
    Test resizing of a 3D image tensor.

    Args:
        test_tif_tensor (torch.Tensor): Resized 3D tensor with stacked bands from the fixture.
    """
    new_height, new_width = 1024, 1024
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_3d)
    assert resized_image.shape == (2, new_height, new_width), f"Unexpected shape: {test_image_3d.shape}"
