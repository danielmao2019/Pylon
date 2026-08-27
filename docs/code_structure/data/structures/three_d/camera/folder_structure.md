# Camera Data Structure Folder Structure

## Code folder structure

```text
data/structures/three_d/camera/
├── __init__.py  # package API surface (re-exports Camera / Cameras + free functions)
├── camera.py    # the Camera class: a CameraIntrinsics + a CameraExtrinsics, plus name / id / device
├── cameras.py   # the Cameras class: an ordered collection / trajectory mirroring the two-object structure over a batch
├── intrinsics/  # the camera-model (intrinsics) subpackage: "what the camera is"
│   ├── __init__.py           # intrinsics API surface
│   ├── camera_intrinsics.py  # the CameraIntrinsics abstract base + per-model subclasses (SimplePinhole / Pinhole / Ortho) + the build_camera_intrinsics factory
│   ├── conventions.py        # image-plane frame transforms, the intrinsics-side counterpart of the extrinsics subpackage's own, routed through the standard pixel frame
│   ├── scaling.py            # the per-axis rescale every length in a params dict takes, plus resolving the two ways a caller names a target resolution; conventions.py reaches in for the first
│   └── validation.py         # intrinsics-layer validations: the single-entry attributes validator, the camera model, the image-plane frame, the per-model params dispatch, and the invariants those params hold only together
├── extrinsics/  # the camera-pose (extrinsics) subpackage: "where the camera is"
│   ├── __init__.py           # extrinsics API surface (re-exports the rotation subpackage)
│   ├── camera_extrinsics.py  # the CameraExtrinsics class: 4x4 cam2world matrix + extr_convention + pose logic
│   ├── conventions.py        # pose-frame transforms, the extrinsics-side counterpart of the intrinsics subpackage's own
│   ├── validation.py         # extrinsics-layer validations: the single-entry attributes validator, the pose frame, the 4x4 extrinsics, and the rotation matrix
│   └── rotation/             # rotation-representation subpackage
│       ├── __init__.py    # rotation API surface
│       ├── euler.py       # Euler-angle rotations
│       ├── pitch_yaw.py   # pitch / yaw rotations
│       ├── quaternion.py  # quaternion rotations
│       ├── rodrigues.py   # Rodrigues / axis-angle rotations
│       └── zero_roll.py   # zero-roll rotation constraint
├── io.py          # generic Camera / Cameras serialization and I/O helpers
├── camera_vis.py  # camera visualization primitives: Camera / Cameras -> vis payload (center, axes, frustum lines)
├── render_camera.py  # renders camera geometry into image space using Bresenham lines
└── validation.py     # camera-level / parent validations (validate_camera_attributes / validate_cameras_attributes): assert each part's type + the name / id / device attributes, relying on each part's own validation for its internals
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/
├── test_intrinsics.py
├── test_conventions.py
├── test_io.py
└── test_rotation_stabilize_validate_compat.py
```
