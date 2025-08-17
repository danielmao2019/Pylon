#!/usr/bin/env python3
"""Test ResizeMaps with challenging ignore value scenarios."""

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

def test_edge_mixing():
    """Test edge cases where ignore values will definitely be mixed."""
    print("Testing challenging cases where ignore values mix with valid values...")
    
    # Case 1: Checkerboard pattern - alternating ignore/valid values
    print("\n1. Checkerboard pattern test:")
    size = 20
    checkerboard = torch.ones((size, size), dtype=torch.float32) * 5.0
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = -1.0
    
    print(f"   Original: {(checkerboard == -1).sum().item()} ignore pixels out of {checkerboard.numel()}")
    
    resize_transform = ResizeMaps(size=(10, 10), interpolation="bilinear")
    resized = resize_transform(checkerboard)
    
    print(f"   Resized: min={resized.min():.3f}, max={resized.max():.3f}")
    print(f"   Exact -1.0 pixels: {(resized == -1.0).sum().item()}")
    print(f"   Negative pixels: {(resized < 0).sum().item()}")
    print(f"   Corrupted pixels (between -1 and 0): {((resized > -1) & (resized < 0)).sum().item()}")
    
    if ((resized > -1) & (resized < 0)).sum().item() > 0:
        print("   ❌ CORRUPTION DETECTED in checkerboard pattern!")
        corrupted_values = resized[(resized > -1) & (resized < 0)]
        print(f"   Sample corrupted values: {corrupted_values[:5]}")
        return False
    
    # Case 2: Single ignore pixel surrounded by valid values
    print("\n2. Single ignore pixel test:")
    single_ignore = torch.ones((10, 10), dtype=torch.float32) * 8.0
    single_ignore[5, 5] = -1.0  # Single ignore pixel in center
    
    print(f"   Original: center pixel = {single_ignore[5, 5]}, neighbors = {single_ignore[4:7, 4:7]}")
    
    # Resize to larger size to see interpolation effects
    resize_transform = ResizeMaps(size=(20, 20), interpolation="bilinear")
    resized = resize_transform(single_ignore)
    
    print(f"   Resized: min={resized.min():.3f}, max={resized.max():.3f}")
    print(f"   Exact -1.0 pixels: {(resized == -1.0).sum().item()}")
    print(f"   Corrupted pixels (between -1 and 0): {((resized > -1) & (resized < 0)).sum().item()}")
    
    if ((resized > -1) & (resized < 0)).sum().item() > 0:
        print("   ❌ CORRUPTION DETECTED in single pixel test!")
        return False
    
    # Case 3: Stripe pattern
    print("\n3. Stripe pattern test:")
    stripes = torch.ones((20, 20), dtype=torch.float32) * 3.0
    stripes[::2, :] = -1.0  # Every other row is ignore
    
    print(f"   Original: {(stripes == -1).sum().item()} ignore pixels")
    
    resize_transform = ResizeMaps(size=(10, 10), interpolation="bilinear")
    resized = resize_transform(stripes)
    
    print(f"   Resized: min={resized.min():.3f}, max={resized.max():.3f}")
    print(f"   Exact -1.0 pixels: {(resized == -1.0).sum().item()}")
    print(f"   Corrupted pixels (between -1 and 0): {((resized > -1) & (resized < 0)).sum().item()}")
    
    if ((resized > -1) & (resized < 0)).sum().item() > 0:
        print("   ❌ CORRUPTION DETECTED in stripe pattern!")
        corrupted_values = resized[(resized > -1) & (resized < 0)]
        print(f"   Sample corrupted values: {corrupted_values[:5]}")
        return False
    
    print("\n✓ All tests passed - no interpolation corruption detected")
    return True

def test_different_resize_factors():
    """Test different resize factors that might expose issues."""
    print("\n" + "="*60)
    print("Testing different resize factors...")
    
    # Create a pattern guaranteed to cause mixing
    original = torch.zeros((6, 6), dtype=torch.float32)
    original[::2, ::2] = 10.0  # Valid values at even positions
    original[1::2, 1::2] = -1.0  # Ignore values at odd positions
    
    print("Original pattern:")
    print(original)
    
    # Test different resize factors
    for new_size in [(3, 3), (4, 4), (5, 5), (9, 9)]:
        print(f"\nResizing to {new_size}:")
        resize_transform = ResizeMaps(size=new_size, interpolation="bilinear")
        resized = resize_transform(original)
        
        print(f"  Min: {resized.min():.3f}, Max: {resized.max():.3f}")
        corrupted = ((resized > -1) & (resized < 0)).sum().item()
        print(f"  Corrupted pixels: {corrupted}")
        
        if corrupted > 0:
            print(f"  ❌ CORRUPTION at size {new_size}!")
            print("  Resized values:")
            print(f"  {resized}")
            return False
    
    return True

if __name__ == "__main__":
    print("Testing ResizeMaps with challenging ignore value scenarios...")
    print("="*70)
    
    success1 = test_edge_mixing()
    success2 = test_different_resize_factors()
    
    print("\n" + "="*70)
    print("FINAL SUMMARY:")
    if success1 and success2:
        print("✓ All tests passed - ResizeMaps handles ignore values correctly")
        print("  This suggests the issue might be more subtle or context-specific")
    else:
        print("❌ CONFIRMED: ResizeMaps corrupts ignore values with bilinear interpolation!")
        exit(1)