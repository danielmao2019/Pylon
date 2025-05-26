import pytest
from data.transforms.resize.maps import ResizeMaps
import os
import torch
from utils.io.image import _load_image, _load_multispectral_image


@pytest.fixture
def test_image_2d() -> torch.Tensor:
    """
    Fixture to load a 2D PNG image as a PyTorch tensor and validate its shape.

    Returns:
        torch.Tensor: Loaded and validated 2D image tensor.
    """
    filepath = "./data/transforms/resize/test_maps/assets/test_png.png"

    # Ensure the file exists
    assert os.path.isfile(filepath), (
        f"Test image not found at {filepath}. Ensure the file is available for testing."
    )

    # Load the image
    image: torch.Tensor = _load_image(filepath)

    # Validate the image shape
    expected_shape = (1024, 1024)
    assert image.shape == expected_shape, (
        f"Unexpected image shape: {image.shape}, expected {expected_shape}."
    )
    return image


@pytest.fixture
def test_image_3d() -> torch.Tensor:
    """
    Fixture to load 3D multispectral .tif image bands as a PyTorch tensor and validate its shape.

    Returns:
        torch.Tensor: Loaded and validated 3D tensor with stacked bands.
    """
    filepaths = [
        "data/transforms/resize/test_maps/assets/test_tif_1.tif",
        "data/transforms/resize/test_maps/assets/test_tif_2.tif",
    ]

    # Ensure all files exist
    assert all(os.path.isfile(fp) for fp in filepaths), (
        f"Test images not found at {filepaths}. Ensure the files are available for testing."
    )

    # Load the multispectral images
    image: torch.Tensor = _load_multispectral_image(filepaths=filepaths, height=512, width=512)

    # Validate the image shape
    expected_shape = (2, 512, 512)
    assert image.shape == expected_shape, (
        f"Unexpected image shape: {image.shape}, expected {expected_shape}."
    )
    return image


@pytest.fixture
def test_image_bmp() -> torch.Tensor:
    """
    Fixture to load a BMP image as a PyTorch tensor and validate its shape.

    Returns:
        torch.Tensor: Loaded and validated BMP image tensor with RGB channels.
    """
    filepath = "./data/transforms/resize/test_maps/assets/1_A.bmp"

    # Ensure the file exists
    assert os.path.isfile(filepath), (
        f"Test BMP image not found at {filepath}. Ensure the file is available for testing."
    )

    # Load the BMP image
    image: torch.Tensor = _load_image(filepath)

    # Validate the image shape
    expected_shape = (3, 1000, 1900)
    assert image.shape == expected_shape, (
        f"Unexpected BMP image shape: {image.shape}, expected {expected_shape}."
    )
    return image


def test_resize_maps_2d(test_image_2d: torch.Tensor) -> None:
    """
    Test resizing a 2D image tensor using the ResizeMaps utility.

    Args:
        test_image_2d (torch.Tensor): 2D image tensor from the fixture.

    Asserts:
        - The resized tensor has the expected shape.
    """
    # Define the new dimensions for resizing
    new_height, new_width = 256, 256

    # Apply the ResizeMaps operation
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_2d)

    # Assert the resized tensor's shape is as expected
    assert resized_image.shape == (new_height, new_width), (
        f"Unexpected resized shape: {resized_image.shape}, expected {(new_height, new_width)}."
    )


def test_resize_maps_3d(test_image_3d: torch.Tensor) -> None:
    """
    Test resizing a 3D image tensor using the ResizeMaps utility.

    Args:
        test_image_3d (torch.Tensor): 3D image tensor from the fixture.

    Asserts:
        - The resized tensor has the expected shape.
    """
    # Define the new dimensions for resizing
    new_height, new_width = 256, 256

    # Apply the ResizeMaps operation
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_3d)

    # Assert the resized tensor's shape is as expected
    expected_shape = (2, new_height, new_width)
    assert resized_image.shape == expected_shape, (
        f"Unexpected resized shape: {resized_image.shape}, expected {expected_shape}."
    )


def test_resize_maps_bmp(test_image_bmp: torch.Tensor) -> None:
    """
    Test resizing a BMP image tensor using the ResizeMaps utility.

    Args:
        test_image_bmp (torch.Tensor): BMP image tensor from the fixture.

    Asserts:
        - The resized tensor has the expected shape.
    """
    # Define the new dimensions for resizing
    new_height, new_width = 256, 256

    # Apply the ResizeMaps operation
    resize_op = ResizeMaps(size=(new_height, new_width))
    resized_image = resize_op(test_image_bmp)

    # Assert the resized tensor's shape is as expected
    expected_shape = (3, new_height, new_width)
    assert resized_image.shape == expected_shape, (
        f"Unexpected resized shape: {resized_image.shape}, expected {expected_shape}."
    )
