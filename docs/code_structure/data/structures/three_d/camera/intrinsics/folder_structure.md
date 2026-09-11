# Camera Intrinsics Folder Structure

## Code folder structure

```text
data/structures/three_d/camera/intrinsics/
├── __init__.py           # intrinsics API surface
├── camera_intrinsics.py  # the CameraIntrinsics abstract base + per-model subclasses (SimplePinhole / Pinhole / Ortho) + the build_camera_intrinsics factory, beside the target-resolution resolver scale_intrinsics reads
├── conventions.py        # image-plane frame transforms, the intrinsics-side counterpart of the extrinsics subpackage's own, routed through the standard pixel frame, with the per-axis rescale each frame's helpers end in
└── validation.py         # intrinsics-layer validations: the single-entry attributes validator, the camera model, the image-plane frame, the per-model params dispatch, and the invariants those params hold only together
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/intrinsics/
└── test_intrinsics.py
```
