# NerfStudio Folder Structure

## Code folder structure

```text
data/structures/three_d/nerfstudio/
├── __init__.py
├── convert.py  # NerfStudio -> COLMAP: the cameras, the images, and the point cloud COLMAP records as points3D
├── load.py  # transforms.json -> the intrinsics, the poses, the modalities, and the split filename lists of one NerfStudio capture
├── nerfstudio_data.py
├── save.py
├── transform.py
└── validate.py
```

## Tests folder structure

```text
tests/data/structures/three_d/nerfstudio/
```
