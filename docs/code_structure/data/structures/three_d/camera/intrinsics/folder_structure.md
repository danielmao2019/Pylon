# Camera Intrinsics Folder Structure

## Code folder structure

```text
data/structures/three_d/camera/intrinsics/
├── __init__.py           # intrinsics API surface
├── camera_intrinsics.py  # the CameraIntrinsics abstract base + per-model subclasses (SimplePinhole / Pinhole / Ortho) + the build_camera_intrinsics factory
├── conventions.py        # image-plane frame transforms, the intrinsics-side counterpart of the extrinsics subpackage's own, routed through the standard pixel frame
├── scaling.py            # the per-axis rescale every length in a params dict takes, plus resolving the two ways a caller names a target resolution; conventions.py reaches in for the first
└── validation.py         # intrinsics-layer validations: the single-entry attributes validator, the camera model, the image-plane frame, the per-model params dispatch, and the invariants those params hold only together
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/intrinsics/
└── test_intrinsics.py
```
