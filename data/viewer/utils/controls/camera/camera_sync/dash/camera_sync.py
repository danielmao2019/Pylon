"""Dash camera-sync store and callback helpers."""

from typing import Any, Dict, List, Optional, Tuple, Union

from dash import Dash, Input, Output, State, dcc, no_update
from dash.development.base_component import Component

from data.viewer.utils.controls.camera.camera_state.dash.camera_state import CameraState

DEFAULT_CAMERA_SYNC_STORE_ID = "camera-sync-store"


def create_camera_sync_store(
    store_id: str = DEFAULT_CAMERA_SYNC_STORE_ID,
) -> Component:
    """Create a Dash store for synchronized camera state.

    Args:
        store_id: Dash component id for the camera-sync store.

    Returns:
        Dash store component initialized with empty camera-sync state.
    """
    assert isinstance(store_id, str), (
        "Store id must be a string. store_id=%r" % store_id
    )
    assert store_id, "Store id must be non-empty. store_id=%r" % store_id
    return dcc.Store(
        id=store_id,
        data={"camera_state": None, "source_id": None, "target_ids": []},
        storage_type="memory",
    )


def register_camera_sync_callbacks(
    app: Dash,
    source_camera_input: Input,
    source_id: Union[str, Dict[str, Any]],
    target_camera_outputs: List[Output],
    store_id: str = DEFAULT_CAMERA_SYNC_STORE_ID,
) -> None:
    """Register Dash callbacks that sync one source camera to target displays.

    Args:
        app: Dash app receiving the callback registration.
        source_camera_input: Single Dash input carrying the latest source camera.
        source_id: Dash id of the spatial display that produced the source camera.
        target_camera_outputs: Dash outputs that receive the active camera.
        store_id: Dash component id for the camera-sync store.

    Returns:
        None.
    """
    assert isinstance(app, Dash), "App must be a Dash instance. app=%r" % (app,)
    assert isinstance(source_camera_input, Input), (
        "Source camera input must be a Dash Input. source_camera_input=%r"
        % source_camera_input
    )
    assert isinstance(source_id, (str, dict)), (
        "Source id must be a string or dict Dash id. source_id=%r" % source_id
    )
    assert isinstance(target_camera_outputs, list), (
        "Target camera outputs must be a list. target_camera_outputs=%r"
        % target_camera_outputs
    )
    for target_camera_output in target_camera_outputs:
        assert isinstance(target_camera_output, Output), (
            "Each target camera output must be a Dash Output. "
            "target_camera_output=%r" % target_camera_output
        )
    assert isinstance(store_id, str), (
        "Store id must be a string. store_id=%r" % store_id
    )
    assert store_id, "Store id must be non-empty. store_id=%r" % store_id

    current_target_ids = [
        target_camera_output.component_id
        for target_camera_output in target_camera_outputs
    ]

    @app.callback(
        [Output(store_id, "data")] + target_camera_outputs,
        [source_camera_input],
        [State(store_id, "data")],
    )
    def sync_camera(
        source_camera: Optional[Dict[str, Any]],
        camera_sync_state: Optional[Dict[str, Any]],
    ) -> List[Any]:
        """Sync the source camera to the registered targets.

        Args:
            source_camera: Latest camera payload from the source display.
            camera_sync_state: Current camera-sync store data.

        Returns:
            Store data followed by one camera value per target output.
        """
        updated_camera_sync_state, target_values = _sync_camera_to_current_targets(
            source_camera=source_camera,
            camera_sync_state=camera_sync_state,
            current_target_ids=current_target_ids,
            source_id=source_id,
        )
        return [updated_camera_sync_state] + target_values

    return None


def _sync_camera_to_current_targets(
    source_camera: Optional[Dict[str, Any]],
    camera_sync_state: Optional[Dict[str, Any]],
    current_target_ids: List[Union[str, Dict[str, Any]]],
    source_id: Union[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Any]]:
    """Sync one active camera to all current non-source targets.

    Args:
        source_camera: Latest camera payload from the source display.
        camera_sync_state: Current camera-sync store data.
        current_target_ids: Dash ids for target spatial displays.
        source_id: Dash id for the source spatial display.

    Returns:
        Updated camera-sync store data and target output values.
    """
    assert source_camera is None or isinstance(source_camera, dict), (
        "Source camera must be None or a dict. source_camera=%r" % source_camera
    )
    assert camera_sync_state is None or isinstance(camera_sync_state, dict), (
        "Camera sync state must be None or a dict. camera_sync_state=%r"
        % camera_sync_state
    )
    assert isinstance(current_target_ids, list), (
        "Current target ids must be a list. current_target_ids=%r" % current_target_ids
    )
    assert isinstance(source_id, (str, dict)), (
        "Source id must be a string or dict Dash id. source_id=%r" % source_id
    )

    updated_camera_sync_state = _set_camera_state_from_source_camera(
        source_camera=source_camera,
        camera_sync_state=camera_sync_state,
        source_id=source_id,
    )
    target_values: List[Any] = []
    for current_target_id in current_target_ids:
        assert isinstance(current_target_id, (str, dict)), (
            "Current target id must be a string or dict Dash id. "
            "current_target_id=%r" % current_target_id
        )
        if current_target_id == source_id:
            target_values.append(no_update)
            continue
        target_values.append(
            apply_camera_state_to_target(
                camera_state=updated_camera_sync_state["camera_state"],
                target_id=current_target_id,
                source_id=source_id,
            )
        )
    return updated_camera_sync_state, target_values


def _set_camera_state_from_source_camera(
    source_camera: Optional[CameraState],
    camera_sync_state: Optional[Dict[str, Any]],
    source_id: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Set active camera state from a source display camera payload.

    Args:
        source_camera: Latest camera payload from the source display.
        camera_sync_state: Current camera-sync store data.
        source_id: Dash id for the source spatial display.

    Returns:
        Updated camera-sync store data.
    """
    assert source_camera is None or isinstance(source_camera, dict), (
        "Source camera must be None or a dict. source_camera=%r" % source_camera
    )
    assert camera_sync_state is None or isinstance(camera_sync_state, dict), (
        "Camera sync state must be None or a dict. camera_sync_state=%r"
        % camera_sync_state
    )
    assert isinstance(source_id, (str, dict)), (
        "Source id must be a string or dict Dash id. source_id=%r" % source_id
    )

    updated_camera_sync_state = (
        {"camera_state": None, "source_id": None, "target_ids": []}
        if camera_sync_state is None
        else dict(camera_sync_state)
    )
    updated_camera_sync_state["camera_state"] = source_camera
    updated_camera_sync_state["source_id"] = source_id
    return updated_camera_sync_state


def apply_camera_state_to_target(
    camera_state: Optional[CameraState],
    target_id: Union[str, Dict[str, Any]],
    source_id: Union[str, Dict[str, Any]],
) -> Any:
    """Apply the active camera value to a Dash spatial-display target.

    Args:
        camera_state: Camera payload to send to the target display.
        target_id: Dash id for the target spatial display.
        source_id: Dash id for the source spatial display.

    Returns:
        Dash callback output value for the target display.
    """
    assert camera_state is None or isinstance(camera_state, dict), (
        "Camera state must be None or a dict. camera_state=%r" % camera_state
    )
    assert isinstance(target_id, (str, dict)), (
        "Target id must be a string or dict Dash id. target_id=%r" % target_id
    )
    assert isinstance(source_id, (str, dict)), (
        "Source id must be a string or dict Dash id. source_id=%r" % source_id
    )
    if target_id == source_id:
        return no_update
    return camera_state
