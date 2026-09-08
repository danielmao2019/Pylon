# Point Cloud IO Folder Structure

## Code folder structure

```text
data/structures/three_d/point_cloud/io/
├── __init__.py          # package marker; re-exports nothing, since the two entry points are already on the parent package's API surface
├── load_point_cloud.py  # any supported point cloud file -> PointCloud: the per-format readers, the layout each format defines, and the dtype each field enters under
└── save_point_cloud.py  # PointCloud -> file
```

## Tests folder structure

```text
tests/utils/io/point_clouds/
├── load_point_cloud/  # the load_point_cloud API and its per-format readers
│   ├── test_point_cloud_loading.py
│   ├── test_point_cloud_operations.py
│   └── test_precision_handling.py
└── save_point_cloud/  # the save_point_cloud API
    └── test_ply_saving.py
```
