import pytest
from ..maps import ResizeMaps
import os
import torch
from utils.io import _load_image, _load_multispectral_image


@pytest.fixture
def test_image_bmp():
    """
    Fixture to load and resize a 2D image as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 2D tensor.
    """
    filepath = "./data/transforms/resize/test_maps/assets/test_png.png"
    assert os.path.isfile(filepath), \
        f"Test image not found at {filepath}. Ensure the file is available for testing."
    image: torch.Tensor = _load_image(filepath)
    assert image.shape == (1024, 1024), \
        f"Unexpected image shape: {image.shape}, expected (1024, 1024)."
    return image


@pytest.fixture
def test_image_3d():
    """
    Fixture to load and resize 3D .tif image bands as a PyTorch tensor.

    Returns:
        torch.Tensor: Resized 3D tensor with stacked bands.
    """
    filepaths = [
        "data/transforms/resize/test_maps/assets/test_tif_1.tif",
        "data/transforms/resize/test_maps/assets/test_tif_2.tif",
    ]    
    assert all(os.path.isfile(x) for x in filepaths), \
        f"Test images not found at {filepaths}. Ensure the files are available for testing."
    image: torch.Tensor = _load_multispectral_image(filepaths=filepaths, height=512, width=512)
    assert image.shape == (2, 512, 512), \
        f"Unexpected image shape: {image.shape}, expected (2, 512, 512)."
    return image


def test_resize_maps(test__image_bmp):
    """
    Test resizing of a 2D image tensor.

    Args:
        test_image_2d (torch.Tensor): Resized 2D tensor from the fixture.
    """
    new_height, new_width = 256, 256
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_bmp)
    assert resized_image.shape == (new_height, new_width), \
        f"Unexpected resized shape: {resized_image.shape}"


def test_resize_maps_3d(test_image_3d):
    """
    Test resizing of a 3D image tensor.

    Args:
        test_image_3d (torch.Tensor): Resized 3D tensor with stacked bands from the fixture.
    """
    new_height, new_width = 1024, 1024
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_3d)
    assert resized_image.shape == (2, new_height, new_width), \
        f"Unexpected resized shape: {resized_image.shape}"
