#!/usr/bin/env python3
"""Test the new test cases with current ResizeMaps implementation."""

import torch
import torchvision

class BaseTransform:
    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self._call_single(args[0], **kwargs)
        else:
            return [self._call_single(arg, **kwargs) for arg in args]
    
    def _call_single(self, x):
        raise NotImplementedError

class ResizeMaps(BaseTransform):
    """Current implementation that should fail the new tests."""
    
    def __init__(self, **kwargs) -> None:
        super(ResizeMaps, self).__init__()
        if 'interpolation' in kwargs:
            if kwargs['interpolation'] is None:
                pass
            elif kwargs['interpolation'] == "bilinear":
                kwargs['interpolation'] = torchvision.transforms.functional.InterpolationMode.BILINEAR
            elif kwargs['interpolation'] == "nearest":
                kwargs['interpolation'] = torchvision.transforms.functional.InterpolationMode.NEAREST
            else:
                raise ValueError(f"Unsupported interpolation mode: {kwargs['interpolation']}")
        if kwargs.get('interpolation', None):
            self.resize_op = torchvision.transforms.Resize(**kwargs)
        else:
            self.resize_op = None
            self.kwargs = kwargs

    def _call_single(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Unsupported tensor dimensions: {x.ndim}. Expected at least 2D tensors.")
        if not self.resize_op:
            interpolation = (
                torchvision.transforms.functional.InterpolationMode.BILINEAR
                if torch.is_floating_point(x) else
                torchvision.transforms.functional.InterpolationMode.NEAREST
            )
            self.kwargs['interpolation'] = interpolation
            resize_op = torchvision.transforms.Resize(**self.kwargs)
        else:
            resize_op = self.resize_op
        x = x.unsqueeze(-3)
        x = resize_op(x)
        x = x.squeeze(-3)
        return x

def test_resize_maps_ignore_values_bilinear():
    """Test that should FAIL with current implementation."""
    print("Testing bilinear interpolation with ignore values...")
    
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
    assert resized_depth.shape == target_size
    
    # Check for corruption (this should FAIL with current implementation)
    corrupted_pixels = ((resized_depth > ignore_value) & (resized_depth < 0)).sum().item()
    
    print(f"  Original pattern: ignore={ignore_value}, valid={valid_value}")
    print(f"  Resized range: [{resized_depth.min():.3f}, {resized_depth.max():.3f}]")
    print(f"  Corrupted pixels: {corrupted_pixels}")
    
    if corrupted_pixels > 0:
        print(f"  ❌ TEST FAILED AS EXPECTED: Found {corrupted_pixels} corrupted pixels")
        print(f"  Sample corrupted values: {resized_depth[(resized_depth > ignore_value) & (resized_depth < 0)][:5]}")
        return False
    else:
        print(f"  ✓ Unexpected: No corruption found")
        return True

def test_resize_maps_depth_map_realistic():
    """Test realistic depth map scenario."""
    print("\nTesting realistic depth map scenario...")
    
    # Create realistic depth map scenario
    height, width = 20, 20
    ignore_value = -1.0
    
    # Valid depth values between 0.5m and 10m
    torch.manual_seed(42)  # For reproducible test
    depth_map = torch.rand((height, width), dtype=torch.float32) * 9.5 + 0.5
    
    # Add ignore regions (simulating sensor limitations)
    depth_map[5:10, 5:10] = ignore_value    # Central ignore region
    depth_map[0:3, :] = ignore_value        # Top border ignore
    depth_map[:, -2:] = ignore_value        # Right border ignore
    
    print(f"  Original ignore pixels: {(depth_map == ignore_value).sum().item()}")
    print(f"  Original valid pixels: {(depth_map != ignore_value).sum().item()}")
    
    # Resize with bilinear interpolation (typical for depth maps)
    target_size = (15, 15)
    resize_op = ResizeMaps(size=target_size, interpolation="bilinear")
    resized_depth = resize_op(depth_map)
    
    # Check for boundary corruption
    boundary_corrupted = ((resized_depth > ignore_value) & (resized_depth < 0)).sum().item()
    
    print(f"  Resized range: [{resized_depth.min():.3f}, {resized_depth.max():.3f}]")
    print(f"  Boundary corrupted pixels: {boundary_corrupted}")
    
    if boundary_corrupted > 0:
        print(f"  ❌ TEST FAILED AS EXPECTED: Found {boundary_corrupted} boundary corrupted pixels")
        corrupted_values = resized_depth[(resized_depth > ignore_value) & (resized_depth < 0)]
        print(f"  Sample corrupted values: {corrupted_values[:5]}")
        return False
    else:
        print(f"  ✓ Unexpected: No boundary corruption found")
        return True

if __name__ == "__main__":
    print("Testing new test cases with current ResizeMaps implementation...")
    print("These tests should FAIL to demonstrate the issue...")
    print("="*70)
    
    test1_passed = test_resize_maps_ignore_values_bilinear()
    test2_passed = test_resize_maps_depth_map_realistic()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    if not test1_passed and not test2_passed:
        print("✓ Both tests failed as expected - confirms the issue exists")
        print("✓ Test cases are correctly designed to catch the problem")
    elif not test1_passed or not test2_passed:
        print("⚠️ One test failed, one passed - issue might be specific to certain cases")
    else:
        print("❌ Both tests passed unexpectedly - issue might be more subtle")
        print("    May need more challenging test cases")
        
    exit(0)  # Exit successfully even if tests "fail" since that's expected