# Data Viewer Camera Controls Folder Structure

## Code folder structure

```text
data/viewer/utils/controls/camera/
├── camera_state/  # generic serialized viewer camera state shared by spatial viewers
│   ├── dash/
│   │   └── camera_state.py  # Dash/Python CameraState contract
│   └── ts/
│       ├── backend/
│       │   ├── schemas/
│       │   │   └── camera_state.py  # TS backend CameraState schema
│       │   └── camera_state.py  # Camera -> TS backend CameraState conversion
│       └── frontend/
│           └── types.ts  # CameraState interface
├── camera_controls/  # generic trackball 3D viewer camera controls
│   ├── dash/
│   │   └── trackball_camera_controls.py  # trackball controls; left-drag rotate, right-drag pan, wheel zoom
│   └── ts/
│       └── frontend/
│           └── trackball_camera_controls.ts  # trackball controls; left-drag rotate, right-drag pan, wheel zoom
└── camera_sync/  # synchronized viewer-camera state shared across spatial displays
    ├── dash/
    │   └── camera_sync.py  # Dash camera-sync store and callback helpers
    └── ts/
        └── frontend/
            ├── types.ts        # CameraSyncState interface
            └── camera_sync.ts  # generic CameraSyncState store with camera-sync-specific additional APIs
```
