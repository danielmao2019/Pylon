import json
from typing import TYPE_CHECKING, Any, Optional

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_recording_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    """Register camera recording callback."""

    @app.callback(
        Output('record-status', 'children'),
        Input('record-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def record_camera_extrinsics_callback(n_clicks: Optional[int]) -> str:
        camera = viewer.get_camera()
        viewer.recorded_cameras.append(camera.extrinsics.cpu().tolist())

        with open(viewer.record_cameras_filepath, 'w', encoding='utf-8') as handle:
            json.dump(viewer.recorded_cameras, handle, indent=2)

        recorded_count = len(viewer.recorded_cameras)
        filepath = viewer.record_cameras_filepath
        return f"Recorded {recorded_count} camera(s) - Saved to {filepath}"
