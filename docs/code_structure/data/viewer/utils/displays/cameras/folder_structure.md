# Data Viewer Cameras Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/cameras/
├── dash/
│   └── camera_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend camera-display response schema: camera-vis JSON payload URL + empty meta_info
    │   ├── core_camera_display.py  # TS DisplayResponse core for cameras
    │   └── apis.py                 # TS backend camera-display APIs
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS camera-display response interface: camera-vis JSON payload URL + empty meta_info
        └── camera_display.ts
```
