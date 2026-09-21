# Data Viewer Mesh Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/mesh/
├── dash/
│   ├── core_mesh_display.py             # Dash mesh display object core
│   ├── mesh_display_textured_viewer.js  # the three.js viewer source the Dash UV-textured mesh display embeds
│   └── apis.py  # Dash mesh-display APIs
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend mesh-display response schemas: url + meta_info
    │   ├── core_mesh_display.py  # TS DisplayResponse core for meshes
    │   └── apis.py               # TS backend mesh-display APIs
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS mesh-display response interfaces: url + meta_info
        ├── core_mesh_display.ts  # TS mesh UI core
        └── apis.ts               # TS frontend mesh-display APIs
```

## Tests folder structure

```text
tests/data/viewer/utils/displays/mesh_display/
```
