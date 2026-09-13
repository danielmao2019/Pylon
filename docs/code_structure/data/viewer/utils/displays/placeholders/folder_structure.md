# Data Viewer Placeholders Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/placeholders/
├── dash/
│   └── placeholder_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend placeholder-display response schema: message
    │   └── placeholder_display.py
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS placeholder-display response interface: message
        └── placeholder_display.ts
```
