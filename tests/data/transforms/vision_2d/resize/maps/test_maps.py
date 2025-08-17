import pytest
import os
import torch
from data.transforms.vision_2d.resize.maps import ResizeMaps
from utils.io.image import _load_image, _load_multispectral_image


@pytest.fixture
def test_image_2d() -> torch.Tensor:
    """
    Fixture to load a 2D PNG image as a PyTorch tensor and validate its shape.

    Returns:
        torch.Tensor: Loaded and validated 2D image tensor.
    """
    filepath = "./tests/data/transforms/vision_2d/resize/maps/assets/test_png.png"

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
        "./tests/data/transforms/vision_2d/resize/maps/assets/test_tif_1.tif",
        "./tests/data/transforms/vision_2d/resize/maps/assets/test_tif_2.tif",
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
    filepath = "./tests/data/transforms/vision_2d/resize/maps/assets/1_A.bmp"

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


def test_resize_maps_ignore_values_bilinear() -> None:
    """
    Test ResizeMaps handling of ignore values with bilinear interpolation.
    This test verifies that ignore values are preserved during resizing operations
    and not corrupted by interpolation with valid values.
    
    Asserts:
        - Ignore values are preserved (not interpolated with valid values)
        - No intermediate values between ignore_value and valid values are created
        - Shape is correct after resizing
    """
    # Create depth map with ignore values
    ignore_value = -1.0
    valid_value = 10.0
    original_size = (6, 6)
    target_size = (5, 5)
    
    # Create pattern that will cause interpolation mixing with current implementation
    depth_map = torch.zeros(original_size, dtype=torch.float32)
    depth_map[::2, ::2] = valid_value      # Valid values at even positions
    depth_map[1::2, 1::2] = ignore_value   # Ignore values at odd positions
    
    # Apply ResizeMaps with bilinear interpolation
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear")
    resized_depth = resize_op(depth_map)
    
    # Verify shape
    assert resized_depth.shape == target_size, (
        f"Unexpected resized shape: {resized_depth.shape}, expected {target_size}."
    )
    
    # CRITICAL: Verify no interpolation corruption of ignore values
    # Values should only be either the ignore_value or valid values, never in between
    corrupted_pixels = ((resized_depth > ignore_value) & (resized_depth < 0)).sum().item()
    assert corrupted_pixels == 0, (
        f"Found {corrupted_pixels} corrupted pixels with interpolated ignore values. "
        f"Min value: {resized_depth.min():.3f}, Max value: {resized_depth.max():.3f}. "
        f"Ignore values should not be interpolated with valid values."
    )
    
    # Verify only expected values exist (ignore_value or valid range)
    unique_values = torch.unique(resized_depth)
    invalid_values = unique_values[(unique_values != ignore_value) & (unique_values < 0)]
    assert len(invalid_values) == 0, (
        f"Found unexpected negative values: {invalid_values}. "
        f"Only {ignore_value} or positive values should exist."
    )


def test_resize_maps_ignore_values_nearest() -> None:
    """
    Test ResizeMaps handling of ignore values with nearest neighbor interpolation.
    Nearest neighbor should always preserve ignore values correctly.
    
    Asserts:
        - Ignore values are perfectly preserved
        - No intermediate values are created
        - Shape is correct after resizing
    """
    # Create depth map with ignore values
    ignore_value = -1.0
    valid_value = 5.0
    original_size = (10, 10)
    target_size = (7, 7)
    
    # Create checkerboard pattern
    depth_map = torch.ones(original_size, dtype=torch.float32) * valid_value
    for i in range(original_size[0]):
        for j in range(original_size[1]):
            if (i + j) % 2 == 0:
                depth_map[i, j] = ignore_value
    
    # Apply ResizeMaps with nearest neighbor interpolation
    resize_op = ResizeMaps(size=target_size, interpolation="nearest")
    resized_depth = resize_op(depth_map)
    
    # Verify shape
    assert resized_depth.shape == target_size, (
        f"Unexpected resized shape: {resized_depth.shape}, expected {target_size}."
    )
    
    # Verify only original values exist (no interpolation)
    unique_values = torch.unique(resized_depth)
    expected_values = {ignore_value, valid_value}
    found_values = set(unique_values.tolist())
    assert found_values.issubset(expected_values), (
        f"Found unexpected interpolated values: {found_values - expected_values}. "
        f"Nearest neighbor should only preserve original values: {expected_values}."
    )


def test_resize_maps_depth_map_realistic() -> None:
    """
    Test ResizeMaps with realistic depth map scenario.
    Simulates a real depth map with ignore values representing invalid measurements.
    
    Asserts:
        - Ignore regions are handled correctly
        - Valid depth values are processed appropriately
        - No corruption at ignore/valid boundaries
    """
    # Create realistic depth map scenario
    height, width = 20, 20
    ignore_value = -1.0
    
    # Valid depth values between 0.5m and 10m
    depth_map = torch.rand((height, width), dtype=torch.float32) * 9.5 + 0.5
    
    # Add ignore regions (simulating sensor limitations)
    depth_map[5:10, 5:10] = ignore_value    # Central ignore region
    depth_map[0:3, :] = ignore_value        # Top border ignore
    depth_map[:, -2:] = ignore_value        # Right border ignore
    
    original_ignore_pixels = (depth_map == ignore_value).sum().item()
    original_valid_pixels = (depth_map != ignore_value).sum().item()
    
    # Resize with bilinear interpolation (typical for depth maps)
    target_size = (15, 15)
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear")
    resized_depth = resize_op(depth_map)
    
    # Verify shape
    assert resized_depth.shape == target_size, (
        f"Unexpected resized shape: {resized_depth.shape}, expected {target_size}."
    )
    
    # CRITICAL: Verify no interpolation corruption at ignore boundaries
    # No values should exist between ignore_value and 0 (corrupted ignore values)
    boundary_corrupted = ((resized_depth > ignore_value) & (resized_depth < 0)).sum().item()
    assert boundary_corrupted == 0, (
        f"Found {boundary_corrupted} pixels with corrupted ignore values at boundaries. "
        f"Range: [{resized_depth.min():.3f}, {resized_depth.max():.3f}]. "
        f"Interpolation should not mix ignore values with valid depth measurements."
    )
    
    # Verify ignore values are properly handled
    resized_ignore_pixels = (resized_depth == ignore_value).sum().item()
    
    # Should have some ignore pixels preserved (exact count depends on resize algorithm)
    # but importantly, no corrupted intermediate values
    valid_range_pixels = (resized_depth >= 0.5).sum().item()
    exact_ignore_pixels = (resized_depth == ignore_value).sum().item()
    total_expected = valid_range_pixels + exact_ignore_pixels
    
    assert total_expected == resized_depth.numel(), (
        f"Found pixels outside expected ranges. "
        f"Valid pixels: {valid_range_pixels}, Ignore pixels: {exact_ignore_pixels}, "
        f"Total: {total_expected}, Expected: {resized_depth.numel()}. "
        f"All pixels should be either valid measurements or exact ignore values."
    )
