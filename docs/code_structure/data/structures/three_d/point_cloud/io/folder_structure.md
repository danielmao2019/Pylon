# Point Cloud IO Folder Structure

## Code folder structure

```text
data/structures/three_d/point_cloud/io/
├── __init__.py          # io package API surface
├── load_point_cloud.py  # any supported point cloud file -> PointCloud: the per-format readers plus device/dtype placement
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
```
