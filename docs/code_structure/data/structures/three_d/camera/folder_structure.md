# Camera Data Structure Folder Structure

## Code folder structure

`./data/structures/three_d/camera/`

```text
camera/
├── __init__.py        # package API surface (re-exports Camera / Cameras + free functions)
├── camera.py          # the Camera class: a CameraIntrinsics + a CameraExtrinsics, plus name / id / device
├── cameras.py         # the Cameras class: an ordered collection / trajectory mirroring the two-object structure over a batch
├── intrinsics/        # the camera-model (intrinsics) subpackage: "what the camera is"
│   ├── __init__.py            # intrinsics API surface
│   ├── camera_intrinsics.py   # the CameraIntrinsics abstract base + per-model subclasses (SimplePinhole / Pinhole / Ortho) + the build_camera_intrinsics factory
│   ├── scaling.py             # camera-intrinsics scaling utilities
│   └── validation.py          # intrinsics-layer validations: the single-entry attributes validator, the camera model, and the per-model params dispatch
├── extrinsics/        # the camera-pose (extrinsics) subpackage: "where the camera is"
│   ├── __init__.py            # extrinsics API surface
│   ├── camera_extrinsics.py   # the CameraExtrinsics class: 4x4 cam2world matrix + convention + pose logic
│   ├── conventions.py         # coordinate-frame convention transforms, relocated under the extrinsics subpackage
│   └── validation.py          # extrinsics-layer validations: the single-entry attributes validator, the convention, the 4x4 extrinsics, and the rotation matrix
├── io.py              # generic Camera / Cameras serialization and I/O helpers
├── camera_vis.py      # camera visualization primitives: Camera / Cameras -> vis payload (center, axes, frustum lines)
├── render_camera.py   # renders camera geometry into image space using Bresenham lines
├── validation.py      # camera-level / parent validations (validate_camera_attributes / validate_cameras_attributes): assert each part's type + the name / id / device attributes, relying on each part's own validation for its internals
└── rotation/          # rotation-representation subpackage
    ├── __init__.py    # rotation API surface
    ├── euler.py       # Euler-angle rotations
    ├── pitch_yaw.py   # pitch / yaw rotations
    ├── quaternion.py  # quaternion rotations
    ├── rodrigues.py   # Rodrigues / axis-angle rotations
    └── zero_roll.py   # zero-roll rotation constraint
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/
├── test_intrinsics.py
├── test_conventions.py
├── test_io.py
└── test_rotation_stabilize_validate_compat.py
```
