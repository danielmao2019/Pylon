# Camera Data Structure Folder Structure

## Code folder structure

```text
./data/structures/three_d/camera/
├── __init__.py        # package API surface (re-exports Camera / Cameras + free functions)
├── camera.py          # the Camera class: single-camera intrinsics + extrinsics
├── cameras.py         # the Cameras class: an ordered collection / trajectory of Camera instances
├── conventions.py     # coordinate-system transformation utilities for rendering
├── io.py              # generic Camera serialization and I/O helpers
├── camera_vis.py      # camera visualization primitives: Camera / Cameras -> vis payload (center, axes, frustum lines)
├── render_camera.py   # renders camera geometry into image space using Bresenham lines
├── scaling.py         # camera scaling utilities
├── validation.py      # Camera field / convention validation
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
├── test_conventions.py
├── test_io.py
└── test_rotation_stabilize_validate_compat.py
```
