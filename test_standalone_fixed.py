#!/usr/bin/env python3
"""Standalone test of the fixed ResizeMaps implementation."""

from typing import Optional
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
    """FIXED ResizeMaps with proper ignore value handling."""
    
    def __init__(self, ignore_value: Optional[float] = None, **kwargs) -> None:
        super(ResizeMaps, self).__init__()
        
        # Store ignore value (can be None for no special handling)
        self.ignore_value = ignore_value
        
        # Handle interpolation parameter
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
        
        # Get resize operation
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
        
        # Use ignore-value-aware resizing only for bilinear interpolation with ignore_value specified
        if (self.ignore_value is not None and 
            resize_op.interpolation == torchvision.transforms.functional.InterpolationMode.BILINEAR):
            return self._ignore_aware_resize(x, resize_op)
        else:
            # Standard resizing for: no ignore_value, nearest neighbor, or other interpolation modes
            return self._standard_resize(x, resize_op)
    
    def _standard_resize(self, x: torch.Tensor, resize_op: torchvision.transforms.Resize) -> torch.Tensor:
        """Apply standard resizing without ignore value handling."""
        x = x.unsqueeze(-3)
        x = resize_op(x)
        x = x.squeeze(-3)
        return x
    
    def _ignore_aware_resize(self, x: torch.Tensor, resize_op: torchvision.transforms.Resize) -> torch.Tensor:
        """Apply resizing with ignore value protection for bilinear interpolation."""
        # Create valid pixel mask
        import math
        if math.isnan(self.ignore_value):
            valid_mask = ~torch.isnan(x)
        else:
            valid_mask = (x != self.ignore_value)
        
        # If no ignore values present, use standard resizing
        if valid_mask.all():
            return self._standard_resize(x, resize_op)
        
        # Replace ignore values with mean of valid values for interpolation
        valid_pixels = x[valid_mask]
        if len(valid_pixels) > 0:
            fill_value = valid_pixels.mean().item()
        else:
            # All pixels are ignore values - use zero
            fill_value = 0.0
        
        # Create data tensor with ignore values replaced
        x_filled = x.clone()
        x_filled[~valid_mask] = fill_value
        
        # Create mask tensor (1.0 for valid, 0.0 for ignore)
        mask = valid_mask.float()
        
        # Resize both data and mask
        x_filled = x_filled.unsqueeze(-3)
        mask = mask.unsqueeze(-3)
        
        x_resized = resize_op(x_filled)
        mask_resized = resize_op(mask)
        
        x_resized = x_resized.squeeze(-3)
        mask_resized = mask_resized.squeeze(-3)
        
        # Restore ignore values where mask indicates invalid regions
        # Use threshold of 0.5 - pixels are valid if mask > 0.5
        ignore_threshold = 0.5
        ignore_regions = mask_resized <= ignore_threshold
        x_resized[ignore_regions] = self.ignore_value
        
        return x_resized

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
    
    success1 = corrupted_pixels == 0
    if success1:
        print("   ✅ SUCCESS: No interpolation corruption!")
    else:
        print(f"   ❌ FAILURE: Still found {corrupted_pixels} corrupted pixels")
        corrupted_values = resized_depth[(resized_depth > ignore_value) & (resized_depth < 0)]
        print(f"   Corrupted values: {corrupted_values}")
    
    # Test Case 2: Without ignore value (should show original problem)
    print("\n2. Testing without ignore value specified (should show corruption):")
    
    resize_transform_normal = ResizeMaps(size=target_size, interpolation="bilinear")
    resized_normal = resize_transform_normal(depth_map)
    
    print(f"   Resized range: [{resized_normal.min():.3f}, {resized_normal.max():.3f}]")
    corrupted_normal = ((resized_normal > ignore_value) & (resized_normal < 0)).sum().item()
    print(f"   Corrupted pixels: {corrupted_normal}")
    
    if corrupted_normal > 0:
        print("   ✅ Expected: Corruption occurs without ignore value protection")
        print(f"   Sample corrupted values: {resized_normal[(resized_normal > ignore_value) & (resized_normal < 0)][:3]}")
    else:
        print("   ⚠️  Unexpected: No corruption (might be okay for this specific case)")
    
    # Test Case 3: Nearest neighbor (should always work)
    print("\n3. Testing nearest neighbor interpolation:")
    
    resize_transform_nearest = ResizeMaps(size=target_size, interpolation="nearest", ignore_value=ignore_value)
    resized_nearest = resize_transform_nearest(depth_map)
    
    print(f"   Resized range: [{resized_nearest.min():.3f}, {resized_nearest.max():.3f}]")
    corrupted_nearest = ((resized_nearest > ignore_value) & (resized_nearest < 0)).sum().item()
    print(f"   Corrupted pixels: {corrupted_nearest}")
    
    success3 = corrupted_nearest == 0
    if success3:
        print("   ✅ SUCCESS: Nearest neighbor preserves ignore values")
    else:
        print(f"   ❌ FAILURE: Unexpected corruption with nearest neighbor")
    
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
    
    success4 = boundary_corrupted == 0
    if success4:
        print("   ✅ SUCCESS: No boundary corruption in realistic scenario")
    else:
        print(f"   ❌ FAILURE: Found {boundary_corrupted} boundary corrupted pixels")
    
    return success1 and success3 and success4

if __name__ == "__main__":
    success = test_fixed_implementation()
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    if success:
        print("🎉 ALL TESTS PASSED - ResizeMaps fix is working correctly!")
    else:
        print("💥 SOME TESTS FAILED - Fix needs more work")
        exit(1)