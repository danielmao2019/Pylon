# COLMAP Folder Structure

## Code folder structure

```text
data/structures/three_d/colmap/
├── __init__.py
├── colmap_data.py  # COLMAP_Data: one reconstruction held as its cameras, images, points3D, with load / transform / save on it
├── convert.py      # COLMAP -> NerfStudio: the transforms record, beside the sparse point cloud ply the frames are posed against
├── load.py         # the COLMAP record types, plus the readers for the .bin / .txt model files
├── save.py         # the writers that put a COLMAP model back out as .bin / .txt model files
├── transform.py    # a similarity transform carried through a whole COLMAP model, its camera poses, its points
└── validate.py     # the structural checks a COLMAP model's cameras, images, points must pass
```

## Tests folder structure

```text
tests/data/structures/three_d/colmap/
```
