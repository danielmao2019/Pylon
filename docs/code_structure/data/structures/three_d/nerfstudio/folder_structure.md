# NerfStudio Folder Structure

## Code folder structure

```text
data/structures/three_d/nerfstudio/
├── __init__.py
├── convert.py  # NerfStudio -> COLMAP: the cameras, the images, and the point cloud COLMAP records as points3D on its own coordinate width and colour range
├── load.py
├── nerfstudio_data.py
├── save.py
├── transform.py
└── validate.py
```

## Tests folder structure

```text
tests/data/structures/three_d/nerfstudio/
└── test_convert.py  # the coordinate width and the colour range a COLMAP export states
```
