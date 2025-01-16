from typing import List, Optional
import os
import pytest
from PIL import Image
import torch
import rasterio
import torchvision
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

def _load_bands(filepaths: List[str], height: Optional[int], width: Optional[int]) -> torch.Tensor:
    """Load multiple bands from separate .tif files into a single tensor."""
    bands: List[torch.Tensor] = []
    print(filepaths)
    for filepath in filepaths:
        with rasterio.open(os.path.join('data/transforms/resize/test_maps/test_tif', filepath)) as src:
            band = src.read(1)
            if band.dtype == numpy.uint16:
                band = band.astype(numpy.int64)
            band = torch.from_numpy(band)[None, :, :]
            assert band.ndim == 3 and band.shape[0] == 1, f"{band.shape=}"
            if height is not None and width is not None:
                band = torchvision.transforms.functional.resize(band, (height, width))
            bands.append(band)
    return torch.cat(bands, dim=0)  # Cat along channel dimension

@pytest.fixture
def test_image_filepath():
    """
    Fixture for the test image filepath.

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
    Fixture for converting a test image to a 2D PyTorch tensor.

    Args:
        test_image_filepath (str): Path to the test image.

    Returns:
        torch.Tensor: 2D PyTorch tensor.
    """
    image = Image.open(test_image_filepath)
    return _pil2torch(image)


@pytest.fixture
def test_tif_tensor():
    """
    Fixture for loading and resizing bands from .tif files.

    Returns:
        torch.Tensor: Tensor containing stacked bands.
    """
    tif_dir = 'data/transforms/resize/test_maps/test_tif'
    if not os.path.isdir(tif_dir):
        pytest.skip(f"TIF directory not found at {tif_dir}.")
    filepaths = os.listdir(tif_dir)
    if not filepaths:
        pytest.skip(f"No .tif files found in {tif_dir}.")
    height, width = 1024, 1024
    return _load_bands(filepaths, height, width)


def test_2d(test_image_tensor):
    """
    Test resizing of a 2D image tensor.

    Args:
        test_image_tensor (torch.Tensor): 2D input tensor.
    """
    assert test_image_tensor.ndimension() == 2, "Input tensor should be 2D."
    height, width = 1024, 1024
    resize_instance = ResizeMaps(size=(height, width))
    resized_image = resize_instance._call_single_(test_image_tensor, height, width)
    assert resized_image.ndimension() == 2, "Resized tensor should remain 2D."
    assert resized_image.shape == (height, width), f"Unexpected shape: {resized_image.shape}"


def test_3d(test_tif_tensor):
    """
    Test resizing of a 3D image tensor.

    Args:
        test_tif_tensor (torch.Tensor): 3D input tensor with stacked bands.
    """
    assert test_tif_tensor.ndimension() == 3, "Input tensor should be 3D."
    height, width = 1024, 1024
    resize_instance = ResizeMaps(size=(height, width))
    resized_image = resize_instance._call_single_(test_tif_tensor, height, width)
    assert resized_image.ndimension() == 3, "Resized tensor should remain 3D."
    assert resized_image.shape[1:] == (height, width), f"Unexpected shape: {resized_image.shape}"