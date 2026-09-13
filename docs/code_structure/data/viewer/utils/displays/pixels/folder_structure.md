# Data Viewer Pixels Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/pixels/
├── dash/
│   ├── core_pixels_display.py  # Dash pixels display object core
│   └── apis.py                 # Dash pixel-display APIs
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend pixel-display response schemas: url + meta_info
    │   ├── core_pixels_display.py  # TS DisplayResponse core for pixels
    │   └── apis.py                 # TS backend pixel-display APIs
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS pixel-display response interfaces: url + meta_info
        ├── core_pixels_display.ts  # TS pixels UI core
        └── apis.ts                 # TS frontend pixel-display APIs
```
