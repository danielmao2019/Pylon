# Data Viewer Placeholders Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/placeholders/dash/placeholder_display.py`

```text
placeholder_display.py
├── from dash import html
└── def create_placeholder_display
    ├── # Builds the Dash missing-result placeholder display from a message.
    ├── impls assert isinstance(message, str)
    ├── impls display = html.Div(message, className="placeholder-surface")
    └── return display  # the slot's stand-in element
```

### Backend schemas

`data/viewer/utils/displays/placeholders/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class PlaceholderDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "placeholder"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── message    # additional field
```

### Backend

`data/viewer/utils/displays/placeholders/ts/backend/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display_response
    ├── # Creates a placeholder display response standing in for a missing result, carrying the message inline.
    ├── impls builds missing-result placeholder response from message
    └── return
```

### Frontend

`data/viewer/utils/displays/placeholders/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface PlaceholderDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "placeholder"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── message    # additional field
```

`data/viewer/utils/displays/placeholders/ts/frontend/placeholder_display.ts`

```text
placeholder_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PlaceholderDisplayResponse } from "./types/display_response";
└── function renderPlaceholderDisplay({ displayResponse }: { displayResponse: PlaceholderDisplayResponse }): LeafVNode
    ├── # Renders the missing-result placeholder UI from the response's message.
    ├── impls complete missing-result placeholder UI from PlaceholderDisplayResponse.message
    └── return LeafVNode keyed by displayResponse.url
```
