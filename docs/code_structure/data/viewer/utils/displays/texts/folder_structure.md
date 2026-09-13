# Data Viewer Texts Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/texts/
├── dash/
│   └── text_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend text-display response schema: url + text + empty meta_info
    │   └── text_display.py
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS text-display response interface: url + text + empty meta_info
        └── text_display.ts
```
