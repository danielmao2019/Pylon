#!/usr/bin/env python3
"""Test ResizeMaps with bilinear interpolation and ignore values."""

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

def test_bilinear_corruption():
    """Test bilinear interpolation corruption of ignore values."""
    print("Testing bilinear interpolation corruption...")
    
    # Create depth map with clear ignore region
    depth_map = torch.ones((100, 100), dtype=torch.float32) * 10.0  # Valid depth = 10.0
    depth_map[30:70, 30:70] = -1.0  # Large ignore region with value -1
    
    print(f"Original depth map (float32 - will use bilinear):")
    print(f"  Shape: {depth_map.shape}")
    print(f"  Valid depth mean: {depth_map[depth_map != -1].mean():.2f}")
    print(f"  Ignore pixels: {(depth_map == -1).sum().item()}")
    print(f"  Min value: {depth_map.min():.2f}")
    print(f"  Max value: {depth_map.max():.2f}")
    
    # Test with explicit bilinear interpolation
    resize_transform = ResizeMaps(size=(50, 50), interpolation="bilinear")
    resized_depth = resize_transform(depth_map)
    
    print(f"\nResized depth map (bilinear):")
    print(f"  Shape: {resized_depth.shape}")
    print(f"  Min value: {resized_depth.min():.2f}")
    print(f"  Max value: {resized_depth.max():.2f}")
    print(f"  Pixels with exact -1.0: {(resized_depth == -1.0).sum().item()}")
    print(f"  Pixels with values < 0: {(resized_depth < 0).sum().item()}")
    print(f"  Pixels with values between -1 and 0: {((resized_depth > -1) & (resized_depth < 0)).sum().item()}")
    
    if (resized_depth < 0).sum() > 0:
        negative_values = resized_depth[resized_depth < 0]
        print(f"  Range of negative values: {negative_values.min():.3f} to {negative_values.max():.3f}")
        unique_negative = torch.unique(negative_values)
        print(f"  Unique negative values (first 10): {unique_negative[:10]}")
    
    # Check for interpolation corruption
    corrupted_pixels = ((resized_depth > -1) & (resized_depth < 0)).sum().item()
    if corrupted_pixels > 0:
        print(f"\n❌ CONFIRMED PROBLEM: {corrupted_pixels} pixels have corrupted ignore values!")
        print("   Bilinear interpolation mixed ignore values (-1) with valid values (10)")
        return False
    else:
        print("\n✓ No interpolation corruption detected")
        return True

def test_nearest_neighbor():
    """Test nearest neighbor interpolation (should preserve ignore values)."""
    print("\n" + "="*60)
    print("Testing nearest neighbor interpolation...")
    
    # Same depth map
    depth_map = torch.ones((100, 100), dtype=torch.float32) * 10.0
    depth_map[30:70, 30:70] = -1.0
    
    # Test with explicit nearest neighbor interpolation
    resize_transform = ResizeMaps(size=(50, 50), interpolation="nearest")
    resized_depth = resize_transform(depth_map)
    
    print(f"Resized depth map (nearest):")
    print(f"  Shape: {resized_depth.shape}")
    print(f"  Min value: {resized_depth.min():.2f}")
    print(f"  Max value: {resized_depth.max():.2f}")
    print(f"  Pixels with exact -1.0: {(resized_depth == -1.0).sum().item()}")
    print(f"  Pixels with values < 0: {(resized_depth < 0).sum().item()}")
    print(f"  Pixels with values between -1 and 0: {((resized_depth > -1) & (resized_depth < 0)).sum().item()}")
    
    corrupted_pixels = ((resized_depth > -1) & (resized_depth < 0)).sum().item()
    if corrupted_pixels > 0:
        print(f"\n❌ UNEXPECTED: {corrupted_pixels} pixels corrupted with nearest neighbor!")
        return False
    else:
        print("\n✓ Nearest neighbor preserves ignore values correctly")
        return True

if __name__ == "__main__":
    print("Testing ResizeMaps ignore value handling...")
    print("="*60)
    
    # Test bilinear (should show corruption)
    bilinear_ok = test_bilinear_corruption()
    
    # Test nearest neighbor (should be OK)
    nearest_ok = test_nearest_neighbor()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Bilinear interpolation: {'✓ OK' if bilinear_ok else '❌ CORRUPTS IGNORE VALUES'}")
    print(f"Nearest neighbor: {'✓ OK' if nearest_ok else '❌ PROBLEM'}")
    
    if not bilinear_ok:
        print("\n🚨 Critical issue confirmed: ResizeMaps corrupts ignore values with bilinear interpolation!")
        exit(1)