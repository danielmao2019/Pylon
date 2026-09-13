# Data Viewer Tables Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/tables/dash/table_display.py`

```text
table_display.py
├── from dash import dash_table
└── def create_table_display
    ├── # Builds the Dash table display from tabular data.
    ├── impls columns = one column spec per field name, over the sorted set of names the rows carry  # sorting the name set is what fixes column order
    ├── impls display = dash_table.DataTable(columns=columns, data=the rows)
    └── return display  # the slot's table element
```

### Backend schemas

`data/viewer/utils/displays/tables/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TableDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "table"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/tables/ts/backend/table_display.py`

```text
table_display.py
└── def create_table_display_response
    ├── # Creates a table display response from a loadable table resource.
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty table metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/tables/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface TableDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "table"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/tables/ts/frontend/table_display.ts`

```text
table_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { TableDisplayResponse } from "./types/display_response";
└── function renderTableDisplay({ displayResponse }: { displayResponse: TableDisplayResponse }): LeafVNode
    ├── # Renders the complete table-display UI from the table resource URL.
    ├── impls complete table-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```
