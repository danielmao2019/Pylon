# Data Viewer Tables Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/tables/
├── dash/
│   └── table_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend table-display response schema: url + empty meta_info
    │   └── table_display.py
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS table-display response interface: url + empty meta_info
        └── table_display.ts
```
