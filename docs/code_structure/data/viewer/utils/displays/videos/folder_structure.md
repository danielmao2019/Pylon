# Data Viewer Videos Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/videos/
├── dash/
│   └── video_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend video-display response schema: url + empty meta_info
    │   └── video_display.py
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS video-display response interface: url + empty meta_info
        └── video_display.ts
```
