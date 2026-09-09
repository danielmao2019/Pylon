# Vision-3D Transforms Folder Structure

## Code folder structure

```text
data/transforms/vision_3d/
├── __init__.py
├── clamp.py
├── downsample.py
├── estimate_normals.py
├── gaussian_pos_noise.py
├── pcr_translation.py
├── random_plane_crop.py
├── random_point_crop.py
├── random_rigid_transform.py  # RandomRigidTransform: poses the source cloud of a registration pair and adjusts the ground truth to match, at the coordinate width the datapoint carries
├── scale.py
├── shuffle.py
├── uniform_pos_noise.py
├── lidar_simulation_crop/
└── pclod/
```

## Tests folder structure

```text
tests/data/transforms/vision_3d/
├── test_downsample.py
├── test_pcr_translation.py
├── test_random_rigid_transform.py  # the posed triplet stays consistent, one seed reproduces it, and the datapoint's own coordinate width is what comes back
├── test_scale.py
└── clamp/
```
