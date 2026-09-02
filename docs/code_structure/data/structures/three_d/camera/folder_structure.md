# Camera Data Structure Folder Structure

## Code folder structure

```text
data/structures/three_d/camera/
├── __init__.py    # package API surface (re-exports Camera / Cameras + free functions)
├── camera.py      # the Camera class: a CameraIntrinsics + a CameraExtrinsics, plus name / id / device
├── cameras.py     # the Cameras class: an ordered collection / trajectory mirroring the two-object structure over a batch
├── intrinsics/    # the camera-model (intrinsics) subpackage: "what the camera is"
├── extrinsics/    # the camera-pose (extrinsics) subpackage: "where the camera is"
├── io.py          # generic Camera / Cameras serialization and I/O helpers
├── camera_vis.py  # camera visualization primitives: Camera / Cameras -> vis payload (center, axes, frustum lines)
├── render_camera.py  # renders camera geometry into image space using Bresenham lines
└── validation.py     # camera-level / parent validations (validate_camera_attributes / validate_cameras_attributes): assert each part's type + the name / id / device attributes, relying on each part's own validation for its internals
```

## Tests folder structure

```text
tests/data/structures/three_d/camera/
├── intrinsics/  # tests of the intrinsics subpackage
├── extrinsics/  # tests of the extrinsics subpackage
├── test_conventions.py
└── test_io.py
```
