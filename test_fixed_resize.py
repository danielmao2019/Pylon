#!/usr/bin/env python3
"""Test the fixed ResizeMaps implementation."""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from data.transforms.vision_2d.resize.maps import ResizeMaps

def test_fixed_implementation():
    """Test the fixed ResizeMaps with ignore value handling."""
    print("Testing FIXED ResizeMaps implementation...")
    print("="*60)
    
    # Test Case 1: Bilinear with ignore values (should be fixed)
    print("1. Testing bilinear interpolation with ignore values:")
    
    ignore_value = -1.0
    valid_value = 10.0
    original_size = (6, 6)
    target_size = (5, 5)
    
    # Create problematic pattern
    depth_map = torch.zeros(original_size, dtype=torch.float32)
    depth_map[::2, ::2] = valid_value      # Valid values at even positions
    depth_map[1::2, 1::2] = ignore_value   # Ignore values at odd positions
    
    print(f"   Original: ignore={ignore_value}, valid={valid_value}")
    print(f"   Original ignore pixels: {(depth_map == ignore_value).sum().item()}")
    
    # Apply FIXED ResizeMaps with ignore value awareness
    resize_transform = ResizeMaps(size=target_size, interpolation="bilinear", ignore_value=ignore_value)
    resized_depth = resize_transform(depth_map)
    
    print(f"   Resized range: [{resized_depth.min():.3f}, {resized_depth.max():.3f}]")
    print(f"   Resized ignore pixels: {(resized_depth == ignore_value).sum().item()}")
    
    # Check for corruption
    corrupted_pixels = ((resized_depth > ignore_value) & (resized_depth < 0)).sum().item()
    print(f"   Corrupted pixels: {corrupted_pixels}")
    
    if corrupted_pixels == 0:
        print("   ✅ SUCCESS: No interpolation corruption!")
    else:
        print(f"   ❌ FAILURE: Still found {corrupted_pixels} corrupted pixels")
        return False
    
    # Test Case 2: Without ignore value (should work normally)
    print("\n2. Testing without ignore value specified:")
    
    resize_transform_normal = ResizeMaps(size=target_size, interpolation="bilinear")
    resized_normal = resize_transform_normal(depth_map)
    
    print(f"   Resized range: [{resized_normal.min():.3f}, {resized_normal.max():.3f}]")
    corrupted_normal = ((resized_normal > ignore_value) & (resized_normal < 0)).sum().item()
    print(f"   Corrupted pixels: {corrupted_normal}")
    
    if corrupted_normal > 0:
        print("   ✅ Expected: Corruption occurs without ignore value protection")
    else:
        print("   ⚠️  Unexpected: No corruption (might be okay for this specific case)")
    
    # Test Case 3: Nearest neighbor (should always work)
    print("\n3. Testing nearest neighbor interpolation:")
    
    resize_transform_nearest = ResizeMaps(size=target_size, interpolation="nearest", ignore_value=ignore_value)
    resized_nearest = resize_transform_nearest(depth_map)
    
    print(f"   Resized range: [{resized_nearest.min():.3f}, {resized_nearest.max():.3f}]")
    corrupted_nearest = ((resized_nearest > ignore_value) & (resized_nearest < 0)).sum().item()
    print(f"   Corrupted pixels: {corrupted_nearest}")
    
    if corrupted_nearest == 0:
        print("   ✅ SUCCESS: Nearest neighbor preserves ignore values")
    else:
        print(f"   ❌ FAILURE: Unexpected corruption with nearest neighbor")
        return False
    
    # Test Case 4: Realistic depth map
    print("\n4. Testing realistic depth map scenario:")
    
    # Create realistic depth map
    torch.manual_seed(42)
    height, width = 20, 20
    depth_map_real = torch.rand((height, width), dtype=torch.float32) * 9.5 + 0.5
    
    # Add ignore regions
    depth_map_real[5:10, 5:10] = ignore_value    # Central ignore region
    depth_map_real[0:3, :] = ignore_value        # Top border ignore
    
    print(f"   Original ignore pixels: {(depth_map_real == ignore_value).sum().item()}")
    
    # Apply fixed resize
    target_size_real = (15, 15)
    resize_transform_real = ResizeMaps(size=target_size_real, interpolation="bilinear", ignore_value=ignore_value)
    resized_real = resize_transform_real(depth_map_real)
    
    print(f"   Resized range: [{resized_real.min():.3f}, {resized_real.max():.3f}]")
    boundary_corrupted = ((resized_real > ignore_value) & (resized_real < 0)).sum().item()
    print(f"   Boundary corrupted pixels: {boundary_corrupted}")
    
    if boundary_corrupted == 0:
        print("   ✅ SUCCESS: No boundary corruption in realistic scenario")
        return True
    else:
        print(f"   ❌ FAILURE: Found {boundary_corrupted} boundary corrupted pixels")
        return False

if __name__ == "__main__":
    success = test_fixed_implementation()
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    if success:
        print("🎉 ALL TESTS PASSED - ResizeMaps fix is working correctly!")
    else:
        print("💥 SOME TESTS FAILED - Fix needs more work")
        exit(1)