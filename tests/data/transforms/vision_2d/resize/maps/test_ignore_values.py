"""
Tests for ResizeMaps ignore value handling.

This module contains comprehensive tests for the ResizeMaps transform's ability to properly
handle ignore values and prevent interpolation corruption. These tests verify that ignore
values (commonly used in depth maps, segmentation masks, etc.) are preserved during resizing
operations and not corrupted by bilinear interpolation.

Key test scenarios:
- Bilinear interpolation with ignore values (primary corruption case)
- Nearest neighbor interpolation (should naturally preserve ignore values)
- Realistic depth map scenarios with border ignore regions
- Edge cases and boundary conditions
"""

import pytest
import torch
from data.transforms.vision_2d.resize.maps import ResizeMaps


def test_resize_maps_ignore_values_bilinear() -> None:
    """
    Test ResizeMaps handling of ignore values with bilinear interpolation.
    
    This is the critical test that verifies the fix for interpolation corruption.
    Without proper ignore value handling, bilinear interpolation would mix ignore
    values with valid values, creating corrupted intermediate values.
    
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
    
    # Create pattern that will cause interpolation mixing without proper handling
    depth_map = torch.zeros(original_size, dtype=torch.float32)
    depth_map[::2, ::2] = valid_value      # Valid values at even positions
    depth_map[1::2, 1::2] = ignore_value   # Ignore values at odd positions
    
    # Apply ResizeMaps with bilinear interpolation and ignore value awareness
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
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
    tolerance = 1e-5
    # Check for values that are negative but not close to ignore_value (corrupted values)
    invalid_values = unique_values[
        (unique_values < 0) & (torch.abs(unique_values - ignore_value) >= tolerance)
    ]
    assert len(invalid_values) == 0, (
        f"Found unexpected negative values: {invalid_values}. "
        f"Only {ignore_value} (±{tolerance}) or positive values should exist."
    )


def test_resize_maps_ignore_values_nearest() -> None:
    """
    Test ResizeMaps handling of ignore values with nearest neighbor interpolation.
    
    Nearest neighbor interpolation should naturally preserve ignore values correctly
    since it doesn't perform interpolation between pixels.
    
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
    
    # Apply ResizeMaps with nearest neighbor interpolation and ignore value awareness
    resize_op = ResizeMaps(size=target_size, interpolation="nearest", ignore_value=ignore_value)
    resized_depth = resize_op(depth_map)
    
    # Verify shape
    assert resized_depth.shape == target_size, (
        f"Unexpected resized shape: {resized_depth.shape}, expected {target_size}."
    )
    
    # Verify only original values exist (no interpolation)
    unique_values = torch.unique(resized_depth)
    tolerance = 1e-5
    
    # Check each unique value is close to either ignore_value or valid_value
    for val in unique_values:
        is_close_to_ignore = torch.abs(val - ignore_value) < tolerance
        is_close_to_valid = torch.abs(val - valid_value) < tolerance
        assert is_close_to_ignore or is_close_to_valid, (
            f"Found unexpected value {val:.6f}. "
            f"Should be close to {ignore_value} or {valid_value} (tolerance={tolerance})."
        )


def test_resize_maps_depth_map_realistic() -> None:
    """
    Test ResizeMaps with realistic depth map scenario.
    
    Simulates a real depth map with ignore values representing invalid measurements,
    such as those produced by depth sensors with limited range or occlusions.
    
    Asserts:
        - Ignore regions are handled correctly
        - Valid depth values are processed appropriately
        - No corruption at ignore/valid boundaries
    """
    # Create realistic depth map scenario
    height, width = 20, 20
    ignore_value = -1.0
    
    # Valid depth values between 0.5m and 10m (typical sensor range)
    torch.manual_seed(42)  # For reproducible test
    depth_map = torch.rand((height, width), dtype=torch.float32) * 9.5 + 0.5
    
    # Add ignore regions (simulating sensor limitations)
    depth_map[5:10, 5:10] = ignore_value    # Central ignore region (occlusion)
    depth_map[0:3, :] = ignore_value        # Top border ignore (out of range)
    depth_map[:, -2:] = ignore_value        # Right border ignore (sensor edge)
    
    tolerance = 1e-5
    original_ignore_pixels = (torch.abs(depth_map - ignore_value) < tolerance).sum().item()
    original_valid_pixels = (torch.abs(depth_map - ignore_value) >= tolerance).sum().item()
    
    # Resize with bilinear interpolation and ignore value awareness (typical for depth maps)
    target_size = (15, 15)
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
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
    # Should have some ignore pixels preserved (exact count depends on resize algorithm)
    # but importantly, no corrupted intermediate values
    tolerance = 1e-5
    valid_range_pixels = (resized_depth >= 0.5).sum().item()
    close_to_ignore_pixels = (torch.abs(resized_depth - ignore_value) < tolerance).sum().item()
    total_expected = valid_range_pixels + close_to_ignore_pixels
    
    assert total_expected == resized_depth.numel(), (
        f"Found pixels outside expected ranges. "
        f"Valid pixels: {valid_range_pixels}, Ignore pixels (±{tolerance}): {close_to_ignore_pixels}, "
        f"Total: {total_expected}, Expected: {resized_depth.numel()}. "
        f"All pixels should be either valid measurements or close to ignore value."
    )


def test_resize_maps_ignore_values_segmentation_mask() -> None:
    """
    Test ResizeMaps with segmentation mask scenario using ignore index 255.
    
    Tests the common segmentation use case where class 255 represents ignore/unlabeled pixels.
    Uses nearest neighbor interpolation which is standard for segmentation masks.
    
    Asserts:
        - Ignore class (255) is preserved
        - Only valid class indices and ignore index exist in output
        - No interpolated class values are created
    """
    # Create segmentation mask with ignore class
    ignore_class = 255
    height, width = 16, 16
    target_size = (12, 12)
    
    # Create segmentation mask with classes 0, 1, 2 and ignore class 255
    seg_mask = torch.randint(0, 3, (height, width), dtype=torch.uint8)
    
    # Add ignore regions
    seg_mask[0:2, :] = ignore_class        # Top border ignore
    seg_mask[:, 0:2] = ignore_class        # Left border ignore
    seg_mask[6:10, 6:10] = ignore_class    # Central ignore region
    
    tolerance = 1e-5
    original_ignore_pixels = (torch.abs(seg_mask.float() - ignore_class) < tolerance).sum().item()
    
    # Apply ResizeMaps with nearest neighbor (standard for segmentation) and ignore class
    resize_op = ResizeMaps(size=target_size, interpolation="nearest", ignore_value=ignore_class)
    resized_mask = resize_op(seg_mask.float())  # Convert to float for ResizeMaps
    
    # Convert back to int for class checks
    resized_mask = resized_mask.round().to(torch.uint8)
    
    # Verify shape
    assert resized_mask.shape == target_size, (
        f"Unexpected resized shape: {resized_mask.shape}, expected {target_size}."
    )
    
    # Verify only valid classes and ignore class exist
    unique_classes = torch.unique(resized_mask)
    expected_classes = {0, 1, 2, ignore_class}
    found_classes = set(unique_classes.tolist())
    assert found_classes.issubset(expected_classes), (
        f"Found unexpected class values: {found_classes - expected_classes}. "
        f"Only valid classes {expected_classes} should exist."
    )
    
    # Verify ignore class is preserved
    tolerance = 1e-5
    resized_ignore_pixels = (torch.abs(resized_mask - ignore_class) < tolerance).sum().item()
    assert resized_ignore_pixels > 0, (
        f"Ignore class {ignore_class} should be preserved in resized mask. "
        f"Original: {original_ignore_pixels}, Resized: {resized_ignore_pixels}"
    )


def test_resize_maps_ignore_nan_values() -> None:
    """
    Test ResizeMaps handling of NaN ignore values.
    
    Tests the case where NaN (Not a Number) is used as the ignore value,
    which is common in scientific datasets for missing measurements.
    
    Asserts:
        - NaN ignore values are handled correctly
        - No corrupted values between NaN and valid values
        - Valid values remain in expected range
    """
    # Create data with NaN ignore values
    ignore_value = float('nan')
    valid_min, valid_max = 1.0, 5.0
    original_size = (8, 8)
    target_size = (6, 6)
    
    # Create data with valid values
    data_map = torch.rand(original_size, dtype=torch.float32) * (valid_max - valid_min) + valid_min
    
    # Add NaN ignore regions
    data_map[0:2, :] = ignore_value        # Top border ignore
    data_map[3:5, 3:5] = ignore_value      # Central ignore region
    
    original_nan_pixels = torch.isnan(data_map).sum().item()
    
    # Apply ResizeMaps with NaN ignore value
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
    resized_data = resize_op(data_map)
    
    # Verify shape
    assert resized_data.shape == target_size, (
        f"Unexpected resized shape: {resized_data.shape}, expected {target_size}."
    )
    
    # Verify NaN values are preserved
    resized_nan_pixels = torch.isnan(resized_data).sum().item()
    assert resized_nan_pixels > 0, (
        f"NaN ignore values should be preserved. "
        f"Original: {original_nan_pixels}, Resized: {resized_nan_pixels}"
    )
    
    # Verify valid values remain in expected range
    valid_pixels = resized_data[~torch.isnan(resized_data)]
    if len(valid_pixels) > 0:
        assert valid_pixels.min() >= valid_min - 0.1, (  # Small tolerance for interpolation
            f"Valid pixels below expected range: min={valid_pixels.min():.3f}, expected>={valid_min}"
        )
        assert valid_pixels.max() <= valid_max + 0.1, (  # Small tolerance for interpolation
            f"Valid pixels above expected range: max={valid_pixels.max():.3f}, expected<={valid_max}"
        )


def test_resize_maps_no_ignore_value_specified() -> None:
    """
    Test ResizeMaps behavior when no ignore value is specified.
    
    Verifies backward compatibility - when ignore_value is None (default),
    the transform should behave exactly as before without special handling.
    
    Asserts:
        - Standard resizing behavior is maintained
        - No special ignore value processing occurs
        - Shape is correct after resizing
    """
    # Create data that would have ignore values if handled specially
    potential_ignore_value = -1.0
    valid_value = 10.0
    original_size = (6, 6)
    target_size = (4, 4)
    
    # Create pattern with potential ignore values
    data_map = torch.zeros(original_size, dtype=torch.float32)
    data_map[::2, ::2] = valid_value
    data_map[1::2, 1::2] = potential_ignore_value
    
    # Apply ResizeMaps WITHOUT ignore_value specified (backward compatibility)
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear")  # ignore_value=None (default)
    resized_data = resize_op(data_map)
    
    # Verify shape
    assert resized_data.shape == target_size, (
        f"Unexpected resized shape: {resized_data.shape}, expected {target_size}."
    )
    
    # Without ignore value handling, interpolation should occur normally
    # This means we might see intermediate values (which is the original "problematic" behavior)
    # But this is expected when ignore_value is not specified
    unique_values = torch.unique(resized_data)
    
    # Should have more than just the two original values due to interpolation
    assert len(unique_values) >= 2, (
        f"Expected interpolated values when ignore_value not specified, "
        f"but found only {len(unique_values)} unique values: {unique_values}"
    )


def test_resize_maps_all_ignore_values() -> None:
    """
    Test ResizeMaps behavior when all pixels are ignore values.
    
    Edge case testing for when the entire input consists of ignore values.
    
    Asserts:
        - All pixels remain as ignore values
        - No errors or exceptions occur
        - Shape is correct after resizing
    """
    ignore_value = -999.0
    original_size = (4, 4)
    target_size = (6, 6)
    
    # Create data where all pixels are ignore values
    data_map = torch.full(original_size, ignore_value, dtype=torch.float32)
    
    # Apply ResizeMaps with ignore value handling
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
    resized_data = resize_op(data_map)
    
    # Verify shape
    assert resized_data.shape == target_size, (
        f"Unexpected resized shape: {resized_data.shape}, expected {target_size}."
    )
    
    # Verify all pixels remain close to ignore values
    tolerance = 1e-5
    all_close_to_ignore = (torch.abs(resized_data - ignore_value) < tolerance).all().item()
    assert all_close_to_ignore, (
        f"All pixels should remain close to ignore value ({ignore_value} ±{tolerance}). "
        f"Found range: [{resized_data.min():.6f}, {resized_data.max():.6f}]"
    )


def test_resize_maps_no_ignore_values_present() -> None:
    """
    Test ResizeMaps behavior when ignore_value is specified but no ignore values are present.
    
    Tests optimization path where ignore value handling is bypassed when not needed.
    
    Asserts:
        - Standard resizing occurs when no ignore values present
        - Performance optimization is utilized
        - Shape is correct after resizing
    """
    ignore_value = -1.0
    original_size = (6, 6)
    target_size = (4, 4)
    
    # Create data with NO ignore values (all valid)
    data_map = torch.rand(original_size, dtype=torch.float32) * 10.0 + 1.0  # Range [1, 11]
    
    # Verify no ignore values are present
    tolerance = 1e-5
    has_ignore = (torch.abs(data_map - ignore_value) < tolerance).any().item()
    assert not has_ignore, "Test data should not contain ignore values"
    
    # Apply ResizeMaps with ignore value specified (but not present in data)
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
    resized_data = resize_op(data_map)
    
    # Verify shape
    assert resized_data.shape == target_size, (
        f"Unexpected resized shape: {resized_data.shape}, expected {target_size}."
    )
    
    # Verify all values remain in valid range
    assert resized_data.min() >= 1.0, (
        f"Resized values below expected range: min={resized_data.min():.3f}"
    )
    assert resized_data.max() <= 11.0, (
        f"Resized values above expected range: max={resized_data.max():.3f}"
    )
    
    # Should not have any ignore values in output
    tolerance = 1e-5
    has_ignore_output = (torch.abs(resized_data - ignore_value) < tolerance).any().item()
    assert not has_ignore_output, (
        f"Output should not contain ignore values when none were present in input"
    )
