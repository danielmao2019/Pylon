#!/usr/bin/env python3
"""Simple test to verify ResizeMaps current behavior with ignore values."""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from data.transforms.vision_2d.resize.maps import ResizeMaps

def test_current_behavior():
    """Test current ResizeMaps behavior with ignore values."""
    print("Testing current ResizeMaps behavior with ignore values...")
    
    # Create a simple depth map with ignore values
    depth_map = torch.ones((100, 100), dtype=torch.float32) * 5.0  # Valid depth = 5.0
    depth_map[40:60, 40:60] = -1.0  # Ignore region with value -1
    
    print(f"Original depth map:")
    print(f"  Shape: {depth_map.shape}")
    print(f"  Valid depth mean: {depth_map[depth_map != -1].mean():.2f}")
    print(f"  Ignore pixels: {(depth_map == -1).sum().item()}")
    print(f"  Min value: {depth_map.min():.2f}")
    print(f"  Max value: {depth_map.max():.2f}")
    
    # Apply ResizeMaps
    resize_transform = ResizeMaps(size=(50, 50))
    resized_depth = resize_transform(depth_map)
    
    print(f"\nResized depth map:")
    print(f"  Shape: {resized_depth.shape}")
    print(f"  Min value: {resized_depth.min():.2f}")
    print(f"  Max value: {resized_depth.max():.2f}")
    print(f"  Pixels with exact -1.0: {(resized_depth == -1.0).sum().item()}")
    print(f"  Pixels with values < 0: {(resized_depth < 0).sum().item()}")
    print(f"  Range of negative values: {resized_depth[resized_depth < 0].min():.2f} to {resized_depth[resized_depth < 0].max():.2f}")
    
    # Check if interpolation corrupted the ignore values
    if (resized_depth < 0).sum() > 0 and (resized_depth == -1.0).sum() == 0:
        print("\n❌ PROBLEM DETECTED: Interpolation corrupted ignore values!")
        print("   Ignore values were interpolated with valid values, creating intermediate values")
    else:
        print("\n✓ No obvious interpolation corruption detected")

if __name__ == "__main__":
    test_current_behavior()