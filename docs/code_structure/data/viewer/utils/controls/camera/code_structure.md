# Data Viewer Camera Controls Code Structure

## 1. Code structure trees

`data/viewer/utils/controls/camera/camera_state/dash/camera_state.py`

```text
camera_state.py
└── class CameraState
    ├── intrinsics
    ├── extrinsics
    ├── intr_convention
    ├── extr_convention
    ├── name
    └── id
```

### Backend schemas

`data/viewer/utils/controls/camera/camera_state/ts/backend/schemas/camera_state.py`

```text
camera_state.py
├── from pydantic import BaseModel
└── class CameraState(BaseModel)
    ├── # One camera's viewer-side state: its intrinsics, extrinsics and convention, plus the name and id identifying it.
    ├── intrinsics
    ├── extrinsics
    ├── intr_convention
    ├── extr_convention
    ├── name
    └── id
```

### Backend

`data/viewer/utils/controls/camera/camera_state/ts/backend/camera_state.py`

```text
camera_state.py
├── from data.structures.three_d.camera import Camera
├── from data.viewer.utils.controls.camera.camera_state.ts.backend.schemas.camera_state import CameraState
└── def create_camera_state_from_camera
    ├── # preserves Camera intrinsics, extrinsics, intr_convention, extr_convention, name, and id
    ├── impls converts Camera to TS backend CameraState schema
    └── return
```

### Frontend

`data/viewer/utils/controls/camera/camera_state/ts/frontend/types.ts`

```text
types.ts
└── interface CameraState
    ├── intrinsics
    ├── extrinsics
    ├── intr_convention
    ├── extr_convention
    ├── name
    └── id
```

`data/viewer/utils/controls/camera/camera_controls/dash/trackball_camera_controls.py`

```text
trackball_camera_controls.py
├── def create_dash_trackball_camera_controls
│   ├── # Builds and validates the Dash trackball controls that every 3D Dash spatial display must use.
│   ├── calls create_dash_renderer_trackball_camera_controls
│   ├── calls assert_dash_trackball_camera_controls
│   └── return
├── def create_dash_renderer_trackball_camera_controls
│   ├── # Constructs the Dash renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls Dash renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu  # impls-node-one-step:skip
│   └── return
├── def assert_dash_trackball_camera_controls
│   ├── # Validates the constructed Dash controls satisfy every trackball contract by running the mouse-mapping, no-orbit, and no-pose-clamp assertions.
│   ├── calls assert_dash_trackball_mouse_mapping
│   ├── calls assert_dash_no_orbit_camera_controls
│   ├── calls assert_dash_no_camera_pose_clamps
│   └── return
├── def assert_dash_trackball_mouse_mapping
│   ├── # Asserts the Dash controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── raise invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── raise context menu blocks trackball panning
│   └── return
├── def assert_dash_no_orbit_camera_controls
│   ├── # Asserts the Dash controls do not use forbidden orbit-style target-locked camera semantics.
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── raise orbit-style camera controls are forbidden
│   └── return
└── def assert_dash_no_camera_pose_clamps
    ├── # Asserts the Dash controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
    ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
    │   └── raise restricted camera pose controls
    └── return
```

`data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls.ts`

```text
trackball_camera_controls.ts
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45
│   └── # Shared vertical-FOV (degrees) every TS spatial display must construct its THREE.PerspectiveCamera with — 45° is the standard 50mm-equivalent lens FOV, trading perspective realism against off-center foreshortening for the orbit-around-near-scene-content use case this lib targets.
├── interface TrackballCameraControls
│   ├── getCameraState
│   │   └── # serializes the entire camera state (every CameraState field — both intrinsics and extrinsics) into a CameraState
│   ├── applyCameraState
│   │   └── # applies the entire CameraState (every field — both intrinsics and extrinsics) to the underlying camera and controls
│   └── subscribeCameraStateChange
├── function createTrackballCameraControls
│   ├── # Builds, validates, and returns the trackball controls, seeding them from initialCameraState and observing the container's data-camera-state attribute for external sync.
│   ├── calls createRendererTrackballCameraControls
│   ├── calls assertTrackballCameraControls
│   ├── if initialCameraState is not null
│   │   └── calls controls.applyCameraState(initialCameraState)
│   ├── impls MutationObserver on container's `data-camera-state` attribute → controls.applyCameraState(parsed state)
│   └── return
├── function createRendererTrackballCameraControls
│   ├── # Constructs the renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu  # impls-node-one-step:skip
│   └── return
├── function assertTrackballCameraControls
│   ├── # Validates the constructed controls satisfy every trackball contract by running the mouse-mapping, no-orbit, and no-pose-clamp assertions.
│   ├── calls assertTrackballMouseMapping
│   ├── calls assertNoOrbitCameraControls
│   ├── calls assertNoCameraPoseClamps
│   └── return
├── function assertTrackballMouseMapping
│   ├── # Asserts the controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── throw invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── throw context menu blocks trackball panning
│   └── return
├── function assertNoOrbitCameraControls
│   ├── # Asserts the controls do not use forbidden orbit-style target-locked camera semantics.
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── throw orbit-style camera controls are forbidden
│   └── return
└── function assertNoCameraPoseClamps
    ├── # Asserts the controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
    ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
    │   └── throw restricted camera pose controls
    └── return
```

`data/viewer/utils/controls/camera/camera_sync/dash/camera_sync.py`

```text
camera_sync.py
├── def create_camera_sync_store
│   ├── # Creates the Dash store that holds the per-source camera-sync registry keyed by source id.
│   ├── impls creates Dash store holding a mapping from source id to its CameraSyncState entry (source id, target ids, current camera state)
│   └── return
├── def register_camera_sync_callbacks
│   ├── # Registers the Dash callbacks that observe each source display's camera and fan its state out to its targets.
│   ├── calls _sync_camera_to_current_targets
│   └── return
├── def _sync_camera_to_current_targets
│   ├── # Dash callback body that commits the firing source's camera and pushes it to every other target registered under that source.
│   ├── calls _set_camera_state_from_source_camera
│   ├── for each current target id from Dash callback inputs or layout pattern ids registered under the firing source
│   │   ├── if target id is source id
│   │   │   └── continue
│   │   └── calls apply_camera_state_to_target
│   └── return
├── def _set_camera_state_from_source_camera
│   ├── # Commits the firing source display's current camera state into that source's CameraSyncState entry in the store.
│   ├── impls assert source_camera is None or isinstance(source_camera, dict)
│   ├── impls assert camera_sync_state is None or isinstance(camera_sync_state, dict)
│   ├── impls assert isinstance(source_id, (str, dict))
│   ├── if camera_sync_state is None
│   │   └── impls updated_camera_sync_state = {"camera_state": None, "source_id": None, "target_ids": []}
│   ├── else
│   │   └── impls updated_camera_sync_state = dict(camera_sync_state)
│   ├── impls updated_camera_sync_state["camera_state"] = source_camera  # committed even when None
│   ├── impls updated_camera_sync_state["source_id"] = source_id
│   └── return updated_camera_sync_state  # the updated camera-sync store data
└── def apply_camera_state_to_target
    ├── # Applies one source's current camera state to a single registered Dash spatial-display target.
    ├── impls applies the source's CameraSyncState.camera_state to a Dash spatial-display target registered under that source
    └── return
```

`data/viewer/utils/controls/camera/camera_sync/ts/frontend/types.ts`

```text
types.ts
└── interface CameraSyncState
    ├── source_id     # the source this entry belongs to; one CameraSyncState exists per source
    ├── target_ids    # targets registered under this source
    └── camera_state  # this source's current camera state
```

`data/viewer/utils/controls/camera/camera_sync/ts/frontend/camera_sync.ts`

```text
camera_sync.ts
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { CameraSyncState } from "./types";
├── class CameraSyncRegistry
│   ├── # Per-source camera-sync registry: each source_id owns an independent CameraSyncState and target element pool, so apply operations stay confined to their source's own pool.
│   ├── _state_by_source_id    # Record<source_id, CameraSyncState> — per-source CameraSyncState entries
│   ├── _targets_by_source_id  # Record<source_id, Map<target_id, HTMLElement>> — per-source target element registry
│   ├── _listeners             # Array<(camera_sync_state: CameraSyncState) => void>
│   ├── loadCameraSyncState
│   │   ├── # Common API: seeds one source's CameraSyncState entry from a caller-provided camera state.
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: empty, camera_state: the caller-provided CameraState }
│   │   ├── impls sets this._targets_by_source_id[source_id] to a fresh empty Map
│   │   └── return
│   ├── getCameraSyncState
│   │   ├── # Common API: reads the current committed CameraSyncState for the given source.
│   │   └── return this._state_by_source_id[source_id]
│   ├── subscribeCameraSyncState
│   │   ├── # Additional API: registers listeners that fire on every apply with the updated source's CameraSyncState.
│   │   ├── impls appends listener to this._listeners
│   │   └── return a callback that removes listener from this._listeners
│   ├── registerCameraSyncTarget
│   │   ├── # Additional API: registers one display panel as a camera-sync target under a specific source; each source owns its own target pool.
│   │   ├── impls idempotently sets this._targets_by_source_id[source_id].set(target_id, target_element)
│   │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
│   │   ├── calls this._apply_camera_state_to_element(target_element, this._state_by_source_id[source_id].camera_state)
│   │   └── return
│   ├── unregisterCameraSyncTarget
│   │   ├── # Additional API: unregisters one display panel from a source's target set.
│   │   ├── impls idempotently deletes this._targets_by_source_id[source_id].delete(target_id)
│   │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
│   │   └── return
│   ├── applyCameraSyncStateToTargets
│   │   ├── # Additional API: applies a caller-owned CameraState to every target registered under one source.
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: the current target_ids, camera_state: the caller-provided CameraState }
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   └── calls this._apply_camera_state_to_element(target_element, camera_state)
│   │   ├── calls this._emit_camera_sync_state(this._state_by_source_id[source_id])
│   │   └── return
│   ├── applySourceCameraStateToTargets
│   │   ├── # Additional API: ingests camera movement from a source display and propagates it to that source's other registered targets.
│   │   ├── if source_id not in this._targets_by_source_id
│   │   │   └── throw
│   │   ├── impls this._state_by_source_id[source_id] = { target_ids: the current target_ids, camera_state: the source display CameraState }
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   ├── if target_id == source_id
│   │   │   │   └── continue
│   │   │   └── calls this._apply_camera_state_to_element(target_element, camera_state)
│   │   ├── calls this._emit_camera_sync_state(this._state_by_source_id[source_id])
│   │   └── return
│   ├── _apply_camera_state_to_element
│   │   ├── # Writes a CameraState onto an element's `data-camera-state` attribute; mesh / point-cloud display containers observe this attribute and re-apply to their trackball controls.
│   │   └── impls sets target_element.dataset.cameraState to the serialized CameraState (or deletes the attribute when CameraState is null)
│   └── _emit_camera_sync_state
│       ├── # Notifies every subscriber with the just-updated source's CameraSyncState.
│       └── for each listener in this._listeners
│           └── impls listener(camera_sync_state)
└── const cameraSyncRegistry = new CameraSyncRegistry()  # the single document-global registry instance shared by every spatial display in the document; consumers import this instance and call its methods
```
