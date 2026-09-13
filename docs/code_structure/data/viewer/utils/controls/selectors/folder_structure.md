# Data Viewer Selector Controls Folder Structure

## Code folder structure

```text
data/viewer/utils/controls/selectors/
├── dash/
│   └── selector_cascade.py  # Dash cascade selector: the dropdown stack from a SelectorResponse, re-rendered per parent change, each level change completed to a full root-leaf path
└── ts/
    ├── backend/
    │   └── schemas/
    │       └── selector_response.py  # SelectorResponse + SelectionNode schema: one axis's (value, label, children) option tree, plus a tree-builder from an app's (value, label, children) tuples
    └── frontend/
        ├── types/
        │   └── selector_response.ts  # SelectorResponse + SelectionNode interfaces mirroring the backend schema
        ├── selection_path.ts    # generic root-leaf selection-path helper: complete a level change to a full root-leaf path (chosen value + first-child descent to a leaf)
        └── selector_cascade.ts  # reusable cascade renderer: (SelectorResponse, current path, onPathChange) -> the dropdown-stack VNode; each <select> keyed by its option-set identity so a coarser-level change re-mounts it; on change the chosen value is completed to a root-leaf path (via selection_path) before onPathChange
```
