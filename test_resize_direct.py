#!/usr/bin/env python3
"""Direct test of ResizeMaps with ignore values."""

import torch
import torchvision
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

class BaseTransform:
    """Minimal BaseTransform implementation for testing."""
    
    def __call__(self, *args, **kwargs):
        if len(args) == 1:
            return self._call_single(args[0], **kwargs)
        else:
            return [self._call_single(arg, **kwargs) for arg in args]
    
    def _call_single(self, x):
        raise NotImplementedError

class ResizeMaps(BaseTransform):
    """
    Copy of ResizeMaps for direct testing.
    """

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
        # sanity check
        assert isinstance(resize_op.size, tuple)
        assert len(resize_op.size) == 2
        assert all(isinstance(s, int) for s in resize_op.size)
        expected_shape = (*x.shape[:-2], *resize_op.size)  # (..., target_H, target_W)
        assert x.shape == expected_shape, f"Resized tensor shape mismatch: expected {expected_shape}, got {x.shape}"
        # return
        return x

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
    if (resized_depth < 0).sum() > 0:
        print(f"  Range of negative values: {resized_depth[resized_depth < 0].min():.2f} to {resized_depth[resized_depth < 0].max():.2f}")
    
    # Check if interpolation corrupted the ignore values
    if (resized_depth < 0).sum() > 0 and (resized_depth == -1.0).sum() == 0:
        print("\n❌ PROBLEM DETECTED: Interpolation corrupted ignore values!")
        print("   Ignore values were interpolated with valid values, creating intermediate values")
        return False
    elif (resized_depth < 0).sum() > 0 and (resized_depth == -1.0).sum() > 0:
        print("\n⚠️  PARTIAL CORRUPTION: Some ignore values preserved, some corrupted")
        return False
    else:
        print("\n✓ No obvious interpolation corruption detected")
        return True

if __name__ == "__main__":
    result = test_current_behavior()
    if not result:
        sys.exit(1)