# Data Viewer Scene Graphs Display Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/scene_graphs/
├── dash/
│   └── scene_graph_display.py
└── ts/
    ├── backend/
    │   ├── schemas/
    │   │   └── display_response.py  # TS backend scene-graph-display response schema: url + empty meta_info
    │   └── scene_graph_display.py
    └── frontend/
        ├── types/
        │   └── display_response.ts  # TS scene-graph-display response interface: url + empty meta_info
        ├── __init__.py
        └── scene_graph_display.ts
```

## Tests folder structure
