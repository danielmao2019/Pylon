# Camera Extrinsics Folder Structure

## Code folder structure

```text
data/structures/three_d/camera/extrinsics/
├── __init__.py           # extrinsics API surface (re-exports the rotation subpackage)
├── camera_extrinsics.py  # the CameraExtrinsics class: 4x4 cam2world matrix + extr_convention + pose logic
├── conventions.py        # pose-frame transforms, the extrinsics-side counterpart of the intrinsics subpackage's own
├── validation.py         # extrinsics-layer validations: the single-entry attributes validator, the pose frame, the 4x4 extrinsics, and the rotation matrix
└── rotation/             # rotation-representation subpackage
    ├── __init__.py    # rotation API surface
    ├── euler.py       # Euler-angle rotations
    ├── pitch_yaw.py   # pitch / yaw rotations
    ├── quaternion.py  # quaternion rotations
    ├── rodrigues.py   # Rodrigues / axis-angle rotations
    └── zero_roll.py   # zero-roll rotation constraint
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/extrinsics/
└── test_rotation_stabilize_validate_compat.py
```
