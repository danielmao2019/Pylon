# Data Viewer Videos Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/videos/dash/video_display.py`

```text
video_display.py
├── from dash import html
└── def create_video_display(src: Optional[str], title: str) -> html.Div
    ├── # Builds the Dash video display from an optional video source url and a title.
    ├── impls assert src is None or isinstance(src, str)
    ├── impls assert isinstance(title, str)
    ├── if src is None
    │   ├── impls placeholder = html.Div("Placeholder for missing video.", className="placeholder-surface")
    │   └── return placeholder
    ├── impls video = html.Video(src=src, controls=True, title=title)
    ├── impls display = html.Div(video)
    └── return display
```

### Backend schemas

`data/viewer/utils/displays/videos/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class VideoDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "video"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/videos/ts/backend/video_display.py`

```text
video_display.py
└── def create_video_display_response
    ├── # Creates a video display response from a loadable video resource.
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty video metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/videos/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface VideoDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "video"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/videos/ts/frontend/video_display.ts`

```text
video_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { VideoDisplayResponse } from "./types/display_response";
└── function renderVideoDisplay({ displayResponse }: { displayResponse: VideoDisplayResponse }): LeafVNode
    ├── # Renders the complete video-display UI from the video resource URL.
    ├── impls complete video-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```
