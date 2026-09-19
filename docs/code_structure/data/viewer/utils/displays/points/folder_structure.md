# Data Viewer Points Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/points/
├── dash/
│   ├── core_points_display.py  # Dash points display object core
│   └── apis.py                 # Dash point-display APIs
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend point-display response schemas: url + meta_info
    │   ├── core_points_display.py  # TS DisplayResponse core for points
    │   └── apis.py                 # TS backend point-display APIs
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS point-display response interfaces: url + meta_info
        ├── core_points_display.ts  # TS points UI core
        └── apis.ts                 # TS frontend point-display APIs
```

## Tests folder structure

```text
tests/data/viewer/utils/displays/point_cloud_display/
├── test_dash_points_style_args.py  # the Dash point-cloud builders' opt-in point_size / point_color overrides
├── test_point_cloud_display.py
├── test_point_cloud_display_invalid_cases.py
├── test_point_cloud_id_utils.py
├── test_point_cloud_integration.py
├── test_point_cloud_lod.py
├── test_point_cloud_stats.py
├── test_point_cloud_utilities.py
└── test_point_cloud_utilities_invalid_cases.py
```
