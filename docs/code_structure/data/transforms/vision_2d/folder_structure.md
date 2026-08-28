# data/transforms/vision_2d — folder structure

## Code folder structure

```text
data/transforms/vision_2d/
├── __init__.py         # exposes AffineResample, the three sampling modes and the two fill rules, beside every other two-dimensional transform this folder holds
├── affine_resample.py  # AffineResample: the one axis-aligned affine raster resample a caller places a source raster into a canonical square through
├── crop/       # Crop and RandomCrop: the fixed and the sampled window a raster is cut to
├── flip.py     # Flip: the mirroring of a raster about one of its axes
├── normalize/  # NormalizeImage and NormalizeDepth: each modality's own value normalization
├── random_rotation.py  # RandomRotation: the sampled rotation a raster is turned by
├── resize/             # ResizeMaps, ResizeNormals and ResizeBBoxes: each of the three resized on its own terms
└── rotation.py         # Rotation: the fixed rotation a raster is turned by
```

## Tests folder structure

```text
tests/data/transforms/vision_2d/
├── crop/    # Crop and RandomCrop
├── resize/  # ResizeMaps
├── test_affine_resample.py  # AffineResample: the three kernels' supports, the widening one, the two fill rules at the source's boundary, and the integer raster's own levels
├── test_flip.py             # Flip
├── test_random_rotation.py  # RandomRotation
└── test_rotation.py         # Rotation
```
