# Data Viewer Selector Controls Code Structure

## 1. Code structure trees

### Backend schemas

`data/viewer/utils/controls/selectors/ts/backend/schemas/selector_response.py`

```text
selector_response.py
├── from typing import List
├── from pydantic import BaseModel
├── def build_selector_response
│   ├── # Build a SelectorResponse from an app's nested (value, label, children) option tuple — the app owns the tree shape, the lib owns the schema.
│   ├── calls _to_selection_node(option_tree)
│   └── return  # SelectorResponse(root=converted imaginary root)
├── def _to_selection_node
│   ├── # Recursion helper: convert one (value, label, children) tuple into a SelectionNode, recursing into each child tuple.
│   ├── for each child tuple
│   │   └── calls _to_selection_node
│   ├── calls SelectionNode
│   └── return  # a SelectionNode holding its converted children
├── class SelectorResponse(BaseModel)
│   ├── # One selector axis: the imaginary root of its option tree, descended recursively along the selection path to render the cascade.
│   └── root: SelectionNode
└── class SelectionNode(BaseModel)
    ├── # One option node of a selector axis: its value, display label, and child nodes (empty at a leaf), so parentage is the nesting itself.
    ├── value: str
    ├── label: str
    └── children: List[SelectionNode]
```

### Frontend

`data/viewer/utils/controls/selectors/ts/frontend/types/selector_response.ts`

```text
selector_response.ts
├── interface SelectorResponse
│   ├── # One selector axis: the imaginary root of its option tree — mirrors the backend SelectorResponse schema.
│   └── root: SelectionNode
└── interface SelectionNode
    ├── # One option node of a selector axis: value, label, and child nodes (empty at a leaf) — mirrors the backend SelectionNode schema.
    ├── value: string
    ├── label: string
    └── children: SelectionNode[]
```

`data/viewer/utils/controls/selectors/ts/frontend/selection_path.ts`

```text
selection_path.ts
├── import type { SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
└── function completeRootLeafPath({ root, path, level, value }: { root: SelectionNode; path: string[]; level: number; value: string }): string[]
    ├── # Complete a selector level change into a full root-leaf path, resetting every finer level to its first option.
    ├── impls start the path with the prefix up to the chosen level plus the chosen value
    ├── for each deeper level until the descended node has no children
    │   ├── impls append the descended node's first child's value
    │   └── impls descend into that first child
    └── return  # the completed root-leaf path
```

`data/viewer/utils/controls/selectors/ts/frontend/selector_cascade.ts`

```text
selector_cascade.ts
├── import type { ElementVNode, LeafVNode } from "web/reconcile/reconcile";
├── import type { SelectorResponse, SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
├── import { completeRootLeafPath } from "data/viewer/utils/controls/selectors/ts/frontend/selection_path";
├── function renderSelectorCascade({ axisKey, response, path, onPathChange }: { axisKey: string; response: SelectorResponse; path: string[]; onPathChange: (next: string[]) => void }): ElementVNode
│   ├── # Render one selector axis as a cascade of native <select> dropdowns, one per level descended from the response's imaginary root down to a leaf.
│   ├── calls _renderSelectorLevel({ node: response.root, level: 0, axisKey, path, onPathChange })  # collect the per-level <select> leaves from the imaginary root down
│   └── return  # a container ElementVNode wrapping the collected <select> leaves
└── function _renderSelectorLevel({ node, level, axisKey, path, onPathChange }: { node: SelectionNode; level: number; axisKey: string; path: string[]; onPathChange: (next: string[]) => void }): LeafVNode[]
    ├── # Recursion helper: collect the <select> leaves from this level down; the base case (a node with no children) contributes none.
    ├── if node has no children
    │   └── return  # [] — base case: a leaf level adds no dropdown
    ├── impls the <select> is a reconciler leaf keyed `${axisKey}-select-${level}-${path[level-1] ?? "root"}` (its option-set identity) so a coarser-level change re-mounts it with this parent's children
    ├── impls build a native <select> over node's children
    ├── function _onLevelChange [local]
    │   ├── # The <select> change handler: report the completed root-leaf path to onPathChange.
    │   ├── calls completeRootLeafPath
    │   └── calls onPathChange
    ├── calls _onLevelChange  # bound as the <select>'s change listener
    ├── calls _renderSelectorLevel({ node: selectedChild, level: level + 1, axisKey, path, onPathChange })  # recurse into the path-selected child to collect the deeper levels' leaves
    └── return  # [this level's <select> leaf, ...the deeper levels' leaves]
```

`data/viewer/utils/controls/selectors/dash/selector_cascade.py`

```text
selector_cascade.py
├── from typing import List
├── from data.viewer.utils.controls.selectors.ts.backend.schemas.selector_response import SelectorResponse, SelectionNode
├── def render_selector_cascade(response: SelectorResponse, path: List[str])
│   ├── # Render one selector axis as a Dash cascade of dropdowns from a SelectorResponse and the current path: one dropdown per level, descending the imaginary root along the path to a leaf, re-rendered per parent change.
│   ├── calls _render_selector_level
│   └── return  # the dropdown-stack Dash component
├── def _render_selector_level(node: SelectionNode, level: int, path: List[str])
│   ├── # Recursion helper: a Dash dropdown over this node's children, then recurse into the child the path selects, stopping at a leaf.
│   ├── if this node has children
│   │   └── calls _render_selector_level
│   └── return
└── def complete_root_leaf_path(node: SelectionNode, path: List[str])
    ├── # Complete a Dash level change into a full root-leaf path: the chosen value, then each deeper level's first child descended to a leaf.
    ├── for each deeper level until the descended node has no children
    │   ├── impls append the descended node's first child's value
    │   └── impls descend into that first child
    └── return  # the completed root-leaf path
```
