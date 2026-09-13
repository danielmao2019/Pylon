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
