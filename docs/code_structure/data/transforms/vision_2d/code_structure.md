# data/transforms/vision_2d — code structure

## Inheritance Relationships

```text
class BaseTransform
└── class AffineResample
```

## Code implementation structure trees

`data/transforms/vision_2d/affine_resample.py`

```text
affine_resample.py
├── from typing import Optional, Tuple
├── import torch
├── from data.transforms.base_transform import BaseTransform
├── SAMPLING_MODE_BOX = "box"              # str — the kernel of stable half-pixel support
├── SAMPLING_MODE_TRIANGLE = "triangle"    # str — the kernel of stable unit support
├── SAMPLING_MODE_CUBIC = "cubic"          # str — the Keys cubic of support two, the one kernel whose support widens with the reduction
├── FILL_RULE_ZERO = "zero"                # str — the source is read as if surrounded by zeros, so a window falls off to zero as it leaves the source
├── FILL_RULE_RENORMALIZE = "renormalize"  # str — the source is read as a picture with a boundary, so an output pixel is the mean of the part of its footprint inside it
├── _WEIGHT_PRECISION_BITS = 32 - 8 - 2    # int — what an eight-bit sample accumulated in a thirty-two-bit integer leaves for the weights carried against it
├── _AXIS_ALIGNED_TOLERANCE = 1e-4         # float — how far off diagonal a transform's linear part may sit while still being read as separable, matching least-squares similarity residue
└── class AffineResample(BaseTransform)
    ├── # Places a raster onto the grid an axis-aligned affine map carries it to, each output pixel gathering along each axis the source samples its kernel weights.
    ├── def __init__(self, transform: torch.Tensor, output_resolution: Tuple[int, int], sampling_mode: str, fill_rule: str) -> None
    │   ├── # Configures one placement: the map source pixel coordinates travel, the window that map is rasterized over, the kernel weighting the samples each output pixel gathers, and how the source's own boundary is read.
    │   ├── impls assert transform is a (3, 3) homogeneous map whose linear part is diagonal to _AXIS_ALIGNED_TOLERANCE  # one scale and one translation per axis, the only shape this resample gathers under
    │   ├── impls assert each axis of transform carries a nonzero scale  # a valid resample needs a scale-backed image of that axis
    │   ├── impls assert output_resolution is an (H, W) pair of positive integer extents
    │   ├── impls assert sampling_mode is one of the three modes this module declares
    │   ├── impls assert fill_rule is one of the two rules this module declares
    │   ├── impls store transform as an attribute          # where each source coordinate lands
    │   ├── impls store output_resolution as an attribute  # how large that landing window is
    │   ├── impls store sampling_mode as an attribute      # which source samples an output pixel gathers, and under what weights
    │   └── impls store fill_rule as an attribute          # what the part of that footprint lying past the source contributes, which the kernel leaves open
    ├── def _call_single(self, raster: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor   [override]
    │   ├── # Resamples one raster under the configured placement, so the base applies that one placement unchanged to every raster it is handed.
    │   ├── impls read each axis's scale off self.transform's diagonal  # separable, so the two axes gather independently
    │   ├── impls read each axis's translation off self.transform's last column
    │   ├── if the raster's dtype is torch.bool
    │   │   └── impls assert self.sampling_mode == SAMPLING_MODE_BOX  # a boolean raster carries selections, so it is placed under the kernel whose footprint holds one source sample
    │   ├── impls result = the raster this call was handed  # the accumulator each axis's gather replaces
    │   ├── for each axis
    │   │   ├── calls self._axis_gather_weights
    │   │   └── calls self._axis_gather(raster=result)  # -> result; an eight-bit raster is back on its own levels before the second axis reads it
    │   └── return  # (..., out_h, out_w), in the raster's own dtype
    ├── def _axis_gather_weights(self, source_extent: int, output_extent: int, scale: float, translation: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]
    │   ├── # Builds one axis's gather: which source indices each output index reads, and the weight it reads each one under.
    │   ├── impls inverse_scale = 1.0 / scale
    │   ├── impls inverse_translation = -translation * inverse_scale                           # the pair inverting this axis's block of the map returns, so a reference that maps its output grid through an inverted matrix is met on its own arithmetic
    │   ├── impls source_coordinate = inverse_scale * each output index + inverse_translation  # integer indices are pixel centres, so a caller's own centre convention rides in the transform it supplies
    │   ├── calls self._kernel_footprint
    │   ├── impls sample_indices = the integer source indices inside the footprint's half-width around each source_coordinate
    │   ├── impls offsets = (sample_indices - source_coordinate) / widening  # into the kernel's own units, so a widened kernel keeps its own shape
    │   ├── calls self._kernel_weights
    │   ├── if self.fill_rule == FILL_RULE_ZERO
    │   │   ├── impls normalize each output index's weights over its whole footprint  # summing to one over the footprint, the samples past the source extent included
    │   │   └── impls zero the weight of every sample index outside source_extent     # what those samples carry is the zero the source is read as surrounded by
    │   ├── if self.fill_rule == FILL_RULE_RENORMALIZE
    │   │   ├── impls zero the weight of every sample index outside source_extent
    │   │   ├── impls normalize each output index's weights over what that zeroing left standing                                         # an output pixel inside the source is the mean of the part of its footprint it can see
    │   │   ├── impls imaged_lower, imaged_upper = the source's own extent carried onto the output grid through this axis's forward map  # the same test in either direction of the scale
    │   │   └── impls zero every weight of an output index lying outside that imaged extent  # such a pixel carries the boundary-zero average, which carries a reference's crop past its own resized frame
    │   └── return sample_indices, weights
    ├── def _axis_gather(self, raster: torch.Tensor, axis: int, sample_indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor
    │   ├── # Contracts one axis of a raster against that axis's gather, an integer raster in its own fixed point and every other raster in the float64 the weights are carried in.
    │   ├── if the raster's dtype is torch.bool
    │   │   └── return the one sample each output index's footprint holds  # a selection, so the truth value that comes back is one the source already carried
    │   ├── if the raster's dtype is an integer one
    │   │   ├── impls quantized = weights carried onto integers of two to the minus _WEIGHT_PRECISION_BITS
    │   │   ├── impls accumulated = the gathered samples contracted against quantized in that integer, the half-step rounding term added in
    │   │   └── return accumulated shifted back by _WEIGHT_PRECISION_BITS and clamped onto the raster's own levels  # this pass returns the raster's own dtype, so the next axis gathers those same levels
    │   └── return the raster contracted against weights in float64  # the weights' own precision, which a float32 raster is narrowed back onto once both axes have gathered
    ├── def _kernel_footprint(self, scale: float) -> Tuple[float, float]
    │   ├── # Returns this mode's footprint on one axis: the factor its support is widened by, and the half-width it gathers over once widened.
    │   ├── if self.sampling_mode == SAMPLING_MODE_BOX
    │   │   └── return 1.0, 0.5
    │   ├── if self.sampling_mode == SAMPLING_MODE_TRIANGLE
    │   │   └── return 1.0, 1.0
    │   ├── if self.sampling_mode == SAMPLING_MODE_CUBIC
    │   │   ├── impls widening = max(1.0 / abs(scale), 1.0)  # the scale's magnitude is what a mirrored axis widens by, while a magnification gathers the kernel's own support
    │   │   └── return widening, 2.0 * widening              # a reduction gathers the source samples its whole footprint covers
    │   └── assert 0, "Should not reach here."
    └── def _kernel_weights(self, offsets: torch.Tensor) -> torch.Tensor
        ├── # Evaluates this mode's kernel at each sample's offset from its own output pixel's source coordinate, in the kernel's own units.
        ├── if self.sampling_mode == SAMPLING_MODE_BOX
        │   └── return the indicator of offsets inside one half
        ├── if self.sampling_mode == SAMPLING_MODE_TRIANGLE
        │   └── return one minus the offsets' magnitude, clamped below at zero
        ├── if self.sampling_mode == SAMPLING_MODE_CUBIC
        │   └── return the Keys cubic of the offsets' magnitude at a = -0.5
        └── assert 0, "Should not reach here."
```
