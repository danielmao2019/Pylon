# Data Viewer Texts Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/texts/dash/text_display.py`

```text
text_display.py
├── from dash import html
└── def create_text_display
    ├── # Builds the Dash text display from a text string.
    ├── impls assert isinstance(text, str)
    ├── impls display = html.Pre(text, className="text-display")
    └── return display  # the slot's text element, whitespace preserved
```

### Backend schemas

`data/viewer/utils/displays/texts/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TextDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "text"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── text       # additional field
```

### Backend

`data/viewer/utils/displays/texts/ts/backend/text_display.py`

```text
text_display.py
└── def create_text_display_response
    ├── # Creates a text display response carrying the text payload inline.
    ├── impls stores text in TextDisplayResponse.text
    ├── impls sets meta_info to empty text metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/texts/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface TextDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "text"  # common field
    ├── url        # common field
    ├── meta_info  # common field
    └── text       # additional field
```

`data/viewer/utils/displays/texts/ts/frontend/text_display.ts`

```text
text_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { TextDisplayResponse } from "./types/display_response";
└── function renderTextDisplay({ displayResponse }: { displayResponse: TextDisplayResponse }): LeafVNode
    ├── # Renders the complete text-display UI from the response's text field.
    ├── impls complete text-display UI from TextDisplayResponse.text
    └── return LeafVNode keyed by displayResponse.url
```
