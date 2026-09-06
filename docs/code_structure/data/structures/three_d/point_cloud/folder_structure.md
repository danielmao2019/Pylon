# Point Cloud Data Structure Folder Structure

## Code folder structure

```text
data/structures/three_d/point_cloud/
├── __init__.py       # package API surface (re-exports PointCloud, Select, RandomSelect, load_point_cloud, save_point_cloud)
├── point_cloud.py    # the PointCloud class: an xyz field plus arbitrary named per-point fields
├── select.py         # Select: index a point cloud down to the points a fixed index list or tensor names
├── random_select.py  # RandomSelect: draw a random subset by percentage or count, through Select
└── io/  # the point-cloud IO subpackage: the per-format readers and the PLY writer
```

## Tests folder structure

```text
tests/data/structures/three_d/
├── test_point_cloud.py  # the PointCloud class: construction, field access, field validation
└── point_cloud/
    └── test_select_random_select.py  # Select and RandomSelect over a PointCloud
```

```text
tests/utils/point_cloud_ops/
├── test_select.py         # Select over the field shapes and index shapes it accepts
└── test_random_select.py  # RandomSelect over its percentage and count modes
```
