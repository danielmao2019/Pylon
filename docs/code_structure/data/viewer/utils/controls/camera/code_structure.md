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
├── import base64
├── import json
├── import math
├── from pathlib import Path
├── from typing import Any, Dict, Optional, Tuple, Union
├── from uuid import uuid4
├── from dash import ALL, Input, clientside_callback
├── PLOTLY_POSE_CLAMPING_DRAGMODE = "turntable"  # Plotly gl3d dragmode that pins camera.up onto world +Z, and the one a scene naming no dragmode runs
├── PLOTLY_FREE_ROLL_DRAGMODE = "orbit"          # Plotly gl3d dragmode whose left-drag carries camera.up with the drag, leaving roll free
├── PLOTLY_DATA_PROPORTION_ASPECTMODE = "data"   # Plotly gl3d aspectmode that draws every axis at its data's own proportions, so a world direction keeps its angles in the scene's normalized space
├── ROLL_LOCKED_GRAPH_ID_TYPE = "dash-roll-locked-graph"  # type of the pattern-matching component id a roll-locked Plotly gl3d display's dcc.Graph carries, the key the roll-lock callback matches it on
├── ROLL_LOCK_CALLBACK_SCRIPT                    # text of roll_lock.js beside this module, the clientside roll-lock source this module registers
├── def create_dash_trackball_camera_controls(renderer_controls: Optional[str] = None, lock_roll: Optional[Tuple[float, float, float]] = None) -> Union[str, Dict[str, Any]]  # renderer_controls: a renderer's own camera-control JavaScript source, or None for a Plotly gl3d display, whose trackball is Plotly's own; lock_roll: a non-zero (x, y, z) world-space axis of any length, or None for the free trackball
│   ├── # Builds and validates the Dash trackball controls that every 3D Dash spatial display must use.
│   ├── calls create_dash_renderer_trackball_camera_controls(renderer_controls=renderer_controls, lock_roll=lock_roll)
│   ├── calls assert_dash_trackball_camera_controls(controls=controls, lock_roll=lock_roll)
│   └── return controls
├── def create_dash_renderer_trackball_camera_controls(renderer_controls: Optional[str], lock_roll: Optional[Tuple[float, float, float]]) -> Union[str, Dict[str, Any]]
│   ├── # Constructs the Dash renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── if renderer_controls is not None
│   │   └── return renderer_controls  # exactly as they arrived, so a display handing over its own source renders what it rendered before lock_roll existed
│   ├── impls plotly_controls = {"scene": {"dragmode": PLOTLY_FREE_ROLL_DRAGMODE}, "graph_id": None}  # the layout.scene configuration a Plotly gl3d display's figure carries and the component id its dcc.Graph carries, Plotly itself wiring left-button rotation, right-button panning, mouse-wheel zoom, and the suppressed canvas context menu
│   ├── if lock_roll is not None
│   │   ├── impls length = the Euclidean length of lock_roll
│   │   ├── impls axis = lock_roll divided by length, normalized to unit length
│   │   ├── impls plotly_controls["scene"]["aspectmode"] = PLOTLY_DATA_PROPORTION_ASPECTMODE  # the scene keeps world directions, so the camera.up below sits on the lock from the first frame and a data-extent change leaves the lock axis in place
│   │   ├── impls plotly_controls["scene"]["camera"] = {"up": {"x": axis[0], "y": axis[1], "z": axis[2]}}
│   │   └── impls plotly_controls["graph_id"] = {"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": uuid4().hex, "lock_roll": base64.b64encode(json.dumps(axis).encode()).decode()}  # index keeps two roll-locked graphs on one page apart; lock_roll hands the callback this graph's axis, base64 so no id value holds a "." Dash escapes in output ids
│   └── return plotly_controls
├── def assert_dash_trackball_camera_controls(controls: Union[str, Dict[str, Any]], lock_roll: Optional[Tuple[float, float, float]] = None) -> None
│   ├── # Validates the constructed Dash controls satisfy every trackball contract by running the mouse-mapping, no-orbit, no-pose-clamp, and roll-lock assertions.
│   ├── calls assert_dash_trackball_mouse_mapping(controls=controls)
│   ├── calls assert_dash_no_orbit_camera_controls(controls=controls)
│   ├── calls assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
│   ├── calls assert_dash_roll_lock(controls=controls, lock_roll=lock_roll)
│   └── return
├── def assert_dash_trackball_mouse_mapping(controls: Union[str, Dict[str, Any]]) -> None
│   ├── # Asserts the Dash controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
│   ├── if controls are Plotly gl3d controls
│   │   ├── assert their scene configuration names no dragmode other than PLOTLY_FREE_ROLL_DRAGMODE or PLOTLY_POSE_CLAMPING_DRAGMODE, "invalid trackball camera controls ..."  # Plotly wires the three-button mapping and suppresses the context menu natively only under a rotation dragmode
│   │   └── return
│   ├── assert controls map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom, "invalid trackball camera controls ..."
│   ├── assert viewer canvas suppresses the default browser context menu, "context menu blocks trackball panning ..."
│   └── return
├── def assert_dash_no_orbit_camera_controls(controls: Union[str, Dict[str, Any]]) -> None
│   ├── # Asserts the Dash controls do not use forbidden orbit-style target-locked camera semantics.
│   ├── assert controls use no orbit-style target-locked camera semantics, "orbit-style camera controls are forbidden ..."  # three's OrbitControls in a renderer source, a pinned camera.center in a Plotly scene configuration
│   └── return
├── def assert_dash_no_camera_pose_clamps(controls: Union[str, Dict[str, Any]], lock_roll: Optional[Tuple[float, float, float]] = None) -> None
│   ├── # Asserts the Dash controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation beyond the polar band a roll lock costs.
│   ├── assert controls are not Plotly gl3d controls whose scene configuration runs PLOTLY_POSE_CLAMPING_DRAGMODE, by name or by naming no dragmode, "restricted camera pose controls ..."
│   ├── assert controls are Plotly gl3d controls or a renderer source restricting no azimuth angle, target lock, distance bounds, pan, or translation, "restricted camera pose controls ..."
│   ├── assert lock_roll is not None or controls are Plotly gl3d controls or a renderer source restricting neither polar angle nor rotation, "restricted camera pose controls ..."
│   ├── assert lock_roll is None or controls are Plotly gl3d controls or a renderer source not restricting rotation, "roll lock must cost only the roll axis and the polar extremes ..."
│   └── return
├── def assert_dash_roll_lock(controls: Union[str, Dict[str, Any]], lock_roll: Optional[Tuple[float, float, float]] = None) -> None
│   ├── # Asserts roll is held about lock_roll when one is supplied and left free when none is, this module owning no axis of its own.
│   ├── assert lock_roll is None or controls hold the camera right axis perpendicular to lock_roll, "roll-locked camera controls must keep the camera right axis perpendicular to the supplied axis ..."
│   ├── assert lock_roll is None or controls hold the camera up vector on lock_roll's side, "roll-locked camera controls must keep the camera up vector on the supplied axis's side ..."
│   ├── if lock_roll is not None and controls are Plotly gl3d controls
│   │   ├── assert their scene's aspectmode is PLOTLY_DATA_PROPORTION_ASPECTMODE, "roll-locked Plotly controls must draw the scene at its data's own proportions ..."
│   │   ├── assert their graph_id is a ROLL_LOCKED_GRAPH_ID_TYPE id carrying exactly type, a str index and a str lock_roll, "roll-locked Plotly controls must carry the graph id the roll-lock callback matches, with the supplied axis ..."
│   │   ├── impls graph_axis = their graph_id's lock_roll, decoded base64 then JSON
│   │   ├── assert graph_axis is lock_roll normalized, three floats each within 1e-9, "roll-locked Plotly controls must carry the graph id the roll-lock callback matches, with the supplied axis ..."
│   │   ├── calls assert_dash_roll_lock(controls=ROLL_LOCK_CALLBACK_SCRIPT, lock_roll=lock_roll)  # the callback that id is matched by holds the lock only as far as its own source does
│   │   └── return
│   ├── assert lock_roll is not None or controls constrain the camera right axis against no axis, "free trackball camera controls must leave camera roll unconstrained ..."  # a pinned camera.up or a roll-locked graph_id in Plotly controls, the roll-lock vocabulary in a renderer source
│   └── return
└── impls clientside_callback(ROLL_LOCK_CALLBACK_SCRIPT, Input({"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": ALL, "lock_roll": ALL}, "relayoutData"))  # module-load registration of the one roll-lock callback, ahead of every Dash app's server setup, so a roll-locked graph a callback adds after the page loaded is matched like one built into the layout
```

`data/viewer/utils/controls/camera/camera_controls/dash/roll_lock.js`

```text
roll_lock.js
└── function holdRollLockedGraphs(relayoutDataList)  # the one clientside callback trackball_camera_controls.py registers; relayoutDataList: the relayoutData of every graph its ROLL_LOCKED_GRAPH_ID_TYPE pattern matches
    ├── # Holds the camera roll of every roll-locked Plotly gl3d graph on the page about the axis its own component id carries, each time any of them reports a relayout.
    ├── function resolveGraphElementId(graphId) [local]
    │   ├── # Resolves the DOM id Dash renders a pattern-matching component id onto, the way dash-renderer stringifies one.
    │   ├── impls graphElementId = "{" + the keys of graphId sorted, each written as JSON.stringify(key) + ":" + JSON.stringify(graphId[key]), joined by "," + "}"
    │   └── return graphElementId
    ├── function createRollLockCallback(graphElementId, worldAxis) [local]  # graphElementId: the DOM id dcc.Graph renders the graph's component id onto; worldAxis: the unit-length lock axis in the data's own world frame, as an [x, y, z] array
    │   ├── # Builds the lock one graph holds: the callback that holds its gl3d scene's camera roll about worldAxis, as the scene draws it, through every drag.
    │   ├── impls ROLL_LOCK_POLAR_ANGLE_EPSILON = the radians the roll-locked camera stops short of either pole of axis
    │   ├── impls ROLL_LOCK_VIOLATION_EPSILON = the squared distance at or below which a reported up vector already equals the roll-locked one
    │   ├── impls ROLL_LOCK_RADIANS_PER_DRAG_UNIT = the radians the view controller's own trackball turns per unit of the screen-space drag it hands its rotation
    │   ├── impls ROLL_LOCK_SUB_STEP_RADIANS = the largest turn one roll-locked drag keyframe takes from the keyframe before it
    │   ├── impls axis, ROLL_LOCK_FALLBACK_MERIDIAN = the lock axis in the mounted scene's normalized space and the meridian an eye sitting on it is banded onto, both set by resolveSceneAxis each time the lock is held on a scene
    │   ├── function rollLockCallback(relayoutData) [local]
    │   │   ├── # Re-holds the lock on graphElementId's gl3d scene each time the roll-lock callback runs, its first render included.
    │   │   ├── calls resolveMountedScene()
    │   │   ├── if graphElementId's gl3d scene has not mounted yet
    │   │   │   ├── impls window.requestAnimationFrame re-runs rollLockCallback(relayoutData) once the scene mounts
    │   │   │   └── return window.dash_clientside.no_update
    │   │   ├── calls holdSceneRollLock(scene)
    │   │   ├── calls subscribeRollLock(graphDiv)
    │   │   ├── calls applyRollLock(graphDiv, scene.getCamera())
    │   │   └── return window.dash_clientside.no_update
    │   ├── function resolveMountedScene() [local]
    │   │   ├── # Resolves graphElementId's Plotly graph div and its gl3d scene, or null while the scene has not mounted.
    │   │   ├── impls graphDiv = the .js-plotly-plot element inside the wrapper whose DOM id is graphElementId
    │   │   ├── if graphDiv, its full layout, or that layout's gl3d scene does not exist yet
    │   │   │   └── return null
    │   │   ├── impls mountedScene = { graphDiv, scene: graphDiv._fullLayout.scene._scene }
    │   │   └── return mountedScene
    │   ├── function holdSceneRollLock(scene) [local]
    │   │   ├── # Holds the lock on one gl3d scene: on the view controller it turns now, and on each one it builds later from the camera the layout stores, before that controller draws a frame.
    │   │   ├── calls resolveSceneAxis(scene)
    │   │   ├── calls holdRollLock(scene.camera.view)
    │   │   ├── if scene already holds the lock
    │   │   │   └── return
    │   │   ├── impls marks scene as holding the lock
    │   │   ├── impls sceneInitializeGLCamera = the scene's own initializeGLCamera, kept for rollLockedInitializeGLCamera to build through
    │   │   ├── function rollLockedInitializeGLCamera() [local]
    │   │   │   ├── # Builds the scene's camera through its own initializeGLCamera, as a projection switch does, then holds the lock on the new view controller in the same call.
    │   │   │   ├── impls sceneInitializeGLCamera called on scene
    │   │   │   ├── calls holdRollLock(scene.camera.view)
    │   │   │   └── return
    │   │   ├── impls scene.initializeGLCamera = rollLockedInitializeGLCamera
    │   │   ├── impls scenePlot = the scene's own plot, kept for rollLockedPlot to plot through
    │   │   ├── function rollLockedPlot(...plotArgs) [local]  # plotArgs: the arguments a figure update or relayout hands the scene's own plot
    │   │   │   ├── # Re-plots the scene through its own plot, which applies a new aspect ratio and new axis ranges, then holds the lock about the axis that aspect leaves before the scene draws its next frame.
    │   │   │   ├── impls scenePlot called on scene with plotArgs
    │   │   │   ├── calls resolveSceneAxis(scene)
    │   │   │   ├── calls rewriteHeldPose(scene.camera.view)
    │   │   │   └── return
    │   │   ├── impls scene.plot = rollLockedPlot
    │   │   └── return
    │   ├── function resolveSceneAxis(scene) [local]
    │   │   ├── # Resolves the lock axis in the scene's normalized space, where Plotly draws each world axis scaled by its aspect ratio over its range, so worldAxis stays upright on screen under any aspect.
    │   │   ├── impls sceneScale = for each of x, y and z, that axis's scene.fullSceneLayout.aspectratio over the span of that axis's range, the per-axis scale the scene draws world coordinates at
    │   │   ├── calls vectorNormalize(worldAxis scaled componentwise by sceneScale)   → axis
    │   │   ├── impls leastBasis = the world basis vector axis leans on least, set by the branches below
    │   │   ├── if Math.abs(axis[0]) <= Math.abs(axis[1]) && Math.abs(axis[0]) <= Math.abs(axis[2])
    │   │   │   └── impls leastBasis = [1, 0, 0]
    │   │   ├── else if Math.abs(axis[1]) <= Math.abs(axis[2])
    │   │   │   └── impls leastBasis = [0, 1, 0]
    │   │   ├── else
    │   │   │   └── impls leastBasis = [0, 0, 1]
    │   │   ├── calls vectorCross(axis, leastBasis)
    │   │   ├── calls vectorNormalize(that cross product)   → ROLL_LOCK_FALLBACK_MERIDIAN
    │   │   └── return
    │   ├── function holdRollLock(view) [local]
    │   │   ├── # Holds the lock on one gl3d scene's view controller, once per view controller, since a replotted graph arrives with a view controller of its own.
    │   │   ├── if view already holds the lock
    │   │   │   └── return
    │   │   ├── impls marks view as holding the lock
    │   │   ├── for each camera controller in view's controller list  # orbital, turntable and matrix; view's own lookAt writes into every one of them
    │   │   │   └── calls holdControllerRollLock(controller)
    │   │   ├── calls rewriteHeldPose(view)  # a view controller a replot built from a rolled stored camera draws on the lock from its next frame
    │   │   ├── function rollLockedRotate(time, yaw, pitch, roll) [local]  # roll: the pure roll a horizontal wheel scroll hands the rotation, which the lock drops
    │   │   │   ├── # Turns the camera by one drag step, as yaw about axis plus pitch about the camera right axis, written as roll-locked sub-step keyframes.
    │   │   │   ├── impls subStepCount = the fewest sub-steps that keep each one's turn within ROLL_LOCK_SUB_STEP_RADIANS
    │   │   │   ├── impls startTime = view.lastT(), the time of view's newest keyframe  # read once, since each sub-step's view.lookAt advances it
    │   │   │   ├── for each sub-step, at evenly spaced times from view's newest keyframe to time  # the renderer then only interpolates between locked poses a bounded turn apart
    │   │   │   │   ├── impls subStepTime = the sub-step's time, evenly spaced from startTime to time
    │   │   │   │   ├── impls view recalculated at subStepTime
    │   │   │   │   ├── impls eye, center = the eye view holds there, the center view holds there
    │   │   │   │   ├── calls resolveTurnedPose(eye, center, yaw / subStepCount, pitch / subStepCount)
    │   │   │   │   └── impls view.lookAt(subStepTime, the turned eye, center, the turned up)
    │   │   │   └── return
    │   │   ├── impls view.rotate = rollLockedRotate
    │   │   └── return
    │   ├── function holdControllerRollLock(controller) [local]
    │   │   ├── # Wraps one camera controller's lookAt so every pose written into its keyframes goes in roll-locked — a drag step, a relayout, a reset-camera button, a replot, and a rotation-mode switch writing into the newly active controller directly.
    │   │   ├── impls controllerLookAt = the controller's own lookAt, kept for rollLockedLookAt to write through
    │   │   ├── function rollLockedLookAt(time, eye, center, up) [local]  # up: the written up, which the lock re-derives from the view direction and axis
    │   │   │   ├── # Writes one pose into the controller's keyframes on the lock.
    │   │   │   ├── impls controller recalculated at time
    │   │   │   ├── impls fills each of eye and center the caller left null from the controller's own pose at time
    │   │   │   ├── calls resolveHeldEye(eye, center, controller, time)   → heldEye
    │   │   │   ├── calls resolveRollLockedPose(heldEye, center)
    │   │   │   ├── impls keyframeCount = the number of keyframes controller.rotation holds before the write  # a controller without a quaternion rotation holds none
    │   │   │   ├── impls controllerLookAt called on the controller with (time, the roll-locked eye, center, the roll-locked up)
    │   │   │   ├── if that write appended a keyframe to the controller's quaternion rotation  # the orbital controller's; the turntable controller keeps angles, which have no second hemisphere
    │   │   │   │   └── calls alignRotationKeyframeHemisphere(controller.rotation)
    │   │   │   └── return
    │   │   ├── impls controller.lookAt = rollLockedLookAt
    │   │   └── return
    │   ├── function alignRotationKeyframeHemisphere(rotation) [local]
    │   │   ├── # Negates the newest rotation keyframe's quaternion when it sits in the opposite hemisphere from the keyframe before it, so the renderer's componentwise interpolation between the two takes the short way round.
    │   │   ├── if the newest and the previous keyframe quaternions have a non-negative dot product
    │   │   │   └── return
    │   │   ├── impls negates the newest keyframe's four components in rotation's state
    │   │   └── return
    │   ├── function rewriteHeldPose(view) [local]
    │   │   ├── # Writes the pose a view holds back onto the lock about the current axis and discards every keyframe before it, so the renderer draws that pose from its next frame.
    │   │   ├── impls view recalculated at view.lastT(), the time of its newest keyframe
    │   │   ├── impls view.lookAt(view.lastT(), the eye, the center, the up view holds there), written through the controllers' rollLockedLookAt
    │   │   ├── impls view.flush(view.lastT()), discarding every keyframe before the pose just written
    │   │   └── return
    │   ├── function subscribeRollLock(graphDiv) [local]
    │   │   ├── # Subscribes the lock once to graphDiv's plotly_relayout and plotly_afterplot events and to the scenes Plotly mounts inside it, so the camera the layout stores is rewritten to the roll-locked pose the renderer already draws.
    │   │   ├── if graphDiv already carries the subscription
    │   │   │   └── return
    │   │   ├── impls graphDiv.__rollLock = { writing: false, pending: false }  # writing: the in-flight flag applyRollLock holds while its own relayout lands; pending: whether a camera write arrived while it did
    │   │   ├── function rewriteWrittenCamera(eventData) [local]
    │   │   │   ├── # Rewrites the camera one relayout event wrote into the layout onto the lock.
    │   │   │   ├── calls resolveWrittenCamera(graphDiv, eventData)
    │   │   │   ├── if the event wrote no camera
    │   │   │   │   └── return
    │   │   │   ├── calls applyRollLock(graphDiv, writtenCamera)
    │   │   │   └── return
    │   │   ├── impls graphDiv.on("plotly_relayout", rewriteWrittenCamera)
    │   │   ├── function rewriteReplottedCamera() [local]
    │   │   │   ├── # Re-holds the lock after each replot, since a Dash figure update reports its camera to no relayout event and a projection switch rebuilds the scene's view controller from the camera the layout stores.
    │   │   │   ├── calls resolveMountedScene()
    │   │   │   ├── if graphElementId's gl3d scene is not mounted
    │   │   │   │   └── return
    │   │   │   ├── calls holdSceneRollLock(scene)
    │   │   │   ├── calls applyRollLock(graphDiv, graphDiv._fullLayout.scene.camera)  # the full layout this replot just rebuilt
    │   │   │   └── return
    │   │   ├── impls graphDiv.on("plotly_afterplot", rewriteReplottedCamera)
    │   │   ├── function holdRebuiltScene() [local]
    │   │   │   ├── # Holds the lock on a gl3d scene Plotly builds anew inside graphDiv, as a figure update that drops the 3D trace and adds it back does, from a mutation observer that runs before that scene's first animation frame.
    │   │   │   ├── calls resolveMountedScene()
    │   │   │   ├── if graphElementId's gl3d scene is not mounted
    │   │   │   │   └── return
    │   │   │   ├── calls holdSceneRollLock(scene)
    │   │   │   └── return
    │   │   ├── impls new MutationObserver(holdRebuiltScene).observe(graphDiv, { childList: true, subtree: true })  # runs as Plotly.react's synchronous part ends, before the rebuilt scene's first animation frame, where plotly_afterplot runs later
    │   │   └── return
    │   ├── function resolveWrittenCamera(graphDiv, eventData) [local]
    │   │   ├── # Resolves the camera one relayout event wrote into graphDiv's layout, or null when it wrote none.
    │   │   ├── if eventData carries scene.camera whole  # a drag, pan or zoom reports the camera it saved, which the full layout may no longer hold
    │   │   │   └── return eventData["scene.camera"]
    │   │   ├── for each key eventData writes
    │   │   │   └── if key is a scene.camera key path or scene.dragmode  # a key-path write rebuilds the full layout's camera, and a switch to turntable re-seats that camera's up on world +Z
    │   │   │       └── return graphDiv._fullLayout.scene.camera
    │   │   └── return null
    │   ├── async function applyRollLock(graphDiv, camera) [local]  # camera: a Plotly layout camera, its eye, center and up {x, y, z} records
    │   │   ├── # Writes the roll-locked pose back to the graph when the camera it reports sits off the lock.
    │   │   ├── calls recordToVector(camera.eye)   → eye
    │   │   ├── calls recordToVector(camera.center)   → center
    │   │   ├── calls recordToVector(camera.up)   → up
    │   │   ├── calls resolveHeldEye(eye, center, the scene's view, the view's latest keyframe time)   → heldEye
    │   │   ├── calls resolveRollLockedPose(heldEye, center)
    │   │   ├── calls vectorSubtract(up, the roll-locked up)   → upDistance
    │   │   ├── calls vectorDot(upDistance, upDistance)
    │   │   ├── if heldEye is the written eye and up already lies within ROLL_LOCK_VIOLATION_EPSILON of the roll-locked up
    │   │   │   └── return
    │   │   ├── if graphDiv's own roll-lock relayout is still in flight
    │   │   │   ├── impls graphDiv.__rollLock.pending = true, so the camera this write leaves is read back once that relayout lands
    │   │   │   └── return
    │   │   ├── impls graphDiv.__rollLock.writing = true, the in-flight flag held until the relayout below resolves
    │   │   ├── calls vectorToRecord(the roll-locked eye)
    │   │   ├── calls vectorToRecord(the roll-locked up)
    │   │   ├── impls await Plotly.relayout(graphDiv, the roll-locked eye record, the roll-locked up record)
    │   │   ├── impls graphDiv.__rollLock.writing = false, releasing the in-flight flag once that relayout resolves
    │   │   ├── if graphDiv.__rollLock.pending
    │   │   │   ├── impls graphDiv.__rollLock.pending = false
    │   │   │   └── calls applyRollLock(graphDiv, graphDiv._fullLayout.scene.camera)  # the camera the writes that arrived in flight left behind
    │   │   └── return
    │   ├── function resolveHeldEye(eye, center, heldPose, time) [local]  # heldPose: the camera controller a write goes into, or the scene's own view
    │   │   ├── # Resolves the eye the lock holds a written camera from, keeping the eye offset heldPose holds at time when the written eye sits on its center and so names no view direction.
    │   │   ├── calls vectorSubtract(eye, center)   → offset
    │   │   ├── calls vectorDot(offset, offset)
    │   │   ├── if eye and center do not coincide
    │   │   │   └── return eye
    │   │   ├── impls heldPose recalculated at time
    │   │   ├── calls vectorSubtract(the eye heldPose holds, the center heldPose holds)
    │   │   ├── calls vectorAdd(center, that eye offset)   → heldEye
    │   │   └── return heldEye
    │   ├── function resolveTurnedPose(eye, center, yaw, pitch) [local]
    │   │   ├── # Turns a roll-locked pose by one drag step's yaw about axis and pitch about the camera right axis.
    │   │   ├── calls vectorSubtract(eye, center)
    │   │   ├── calls resolveBandedOffset(that eye offset)   → bandedOffset
    │   │   ├── calls vectorRotateAboutAxis(bandedOffset, axis, ROLL_LOCK_RADIANS_PER_DRAG_UNIT × yaw)   → yawedOffset
    │   │   ├── calls vectorScale(yawedOffset, −1)
    │   │   ├── calls vectorCross(that reversed offset, axis)
    │   │   ├── calls vectorNormalize(that cross product)   → right
    │   │   ├── calls vectorDot(yawedOffset, axis)
    │   │   ├── calls vectorDot(yawedOffset, yawedOffset)
    │   │   ├── impls polarAngle = the angle between yawedOffset and axis, from those two dot products
    │   │   ├── impls pitchAngle = −ROLL_LOCK_RADIANS_PER_DRAG_UNIT × pitch, clamped to the polar band ROLL_LOCK_POLAR_ANGLE_EPSILON short of both poles of axis, so a drag stops at a pole instead of carrying the view through it
    │   │   ├── calls vectorRotateAboutAxis(yawedOffset, right, pitchAngle)   → turnedOffset
    │   │   ├── calls vectorScale(turnedOffset, −1)
    │   │   ├── calls vectorCross(right, that reversed offset)
    │   │   ├── calls vectorNormalize(that cross product)   → turnedUp
    │   │   ├── calls vectorAdd(center, turnedOffset)
    │   │   ├── impls turnedPose = { eye: that sum, up: turnedUp }
    │   │   └── return turnedPose
    │   ├── function resolveRollLockedPose(eye, center) [local]  # eye, center: [x, y, z] arrays
    │   │   ├── # Resolves the pose the lock holds a camera at: its eye where it was written, banded off the poles, and the up vector the view direction from that eye and axis determine, whatever up was written.
    │   │   ├── calls vectorSubtract(eye, center)
    │   │   ├── calls resolveBandedOffset(that eye offset)   → bandedOffset
    │   │   ├── calls vectorAdd(center, bandedOffset)   → rollLockedEye
    │   │   ├── calls vectorSubtract(center, rollLockedEye)
    │   │   ├── calls vectorNormalize(that view offset)   → forward
    │   │   ├── calls vectorCross(forward, axis)
    │   │   ├── calls vectorNormalize(that cross product)   → right  # held perpendicular to axis
    │   │   ├── calls vectorCross(right, forward)
    │   │   ├── calls vectorNormalize(that cross product)
    │   │   ├── impls rollLockedPose = { eye: rollLockedEye, up: that unit vector }
    │   │   └── return rollLockedPose
    │   ├── function resolveBandedOffset(offset) [local]
    │   │   ├── # Bands an eye offset's polar angle off axis into [ROLL_LOCK_POLAR_ANGLE_EPSILON, π − ROLL_LOCK_POLAR_ANGLE_EPSILON], rebuilding it at the banded angle on its own meridian.
    │   │   ├── calls vectorDot(offset, offset)
    │   │   ├── impls radius = the length of offset, the square root of that squared length
    │   │   ├── calls vectorDot(offset, axis)
    │   │   ├── impls polarAngle = the angle between offset and axis, from that dot product over radius
    │   │   ├── if polarAngle already lies inside the band
    │   │   │   └── return offset
    │   │   ├── calls resolveMeridian(offset)
    │   │   ├── impls bandedPolarAngle = polarAngle clamped into the band
    │   │   ├── calls vectorScale(axis, radius·cos(bandedPolarAngle))
    │   │   ├── calls vectorScale(meridian, radius·sin(bandedPolarAngle))
    │   │   ├── calls vectorAdd(that axial part, that meridian part)   → bandedOffset
    │   │   └── return bandedOffset
    │   ├── function resolveMeridian(offset) [local]
    │   │   ├── # Resolves the meridian an eye offset stands on, as a unit vector perpendicular to axis.
    │   │   ├── calls vectorDot(offset, axis)
    │   │   ├── calls vectorScale(axis, that dot product)
    │   │   ├── calls vectorSubtract(offset, that axial part)   → meridian
    │   │   ├── calls vectorDot(meridian, meridian)
    │   │   ├── if meridian has zero length  # an offset on axis stands on every meridian at once
    │   │   │   └── return ROLL_LOCK_FALLBACK_MERIDIAN
    │   │   ├── calls vectorNormalize(meridian)
    │   │   └── return unitMeridian
    │   ├── function vectorRotateAboutAxis(vector, unitAxis, angle) [local]
    │   │   ├── # Rotates a vector about a unit axis by an angle in radians, right-handed, the way a drag's yaw and pitch turn the eye offset.
    │   │   ├── impls cosine and sine of angle
    │   │   ├── calls vectorScale(vector, cosine)
    │   │   ├── calls vectorCross(unitAxis, vector)
    │   │   ├── calls vectorScale(that cross product, sine)
    │   │   ├── calls vectorAdd(the scaled vector, the scaled cross product)
    │   │   ├── calls vectorDot(unitAxis, vector)
    │   │   ├── calls vectorScale(unitAxis, that dot product × (1 − cosine))
    │   │   ├── calls vectorAdd(that sum, the scaled unitAxis)   → rotated  # Rodrigues' rotation of vector about unitAxis by angle
    │   │   └── return rotated
    │   ├── function vectorAdd(left, right) [local]
    │   │   ├── # Adds two [x, y, z] arrays.
    │   │   ├── impls sum = the componentwise sum of left and right
    │   │   └── return sum
    │   ├── function vectorSubtract(left, right) [local]
    │   │   ├── # Subtracts one [x, y, z] array from another.
    │   │   ├── impls difference = the componentwise difference of left and right
    │   │   └── return difference
    │   ├── function vectorScale(vector, scalar) [local]
    │   │   ├── # Scales an [x, y, z] array by a scalar, negation included.
    │   │   ├── impls scaled = each component of vector times scalar
    │   │   └── return scaled
    │   ├── function vectorDot(left, right) [local]
    │   │   ├── # Resolves the dot product of two [x, y, z] arrays, a squared length when both are one vector.
    │   │   ├── impls dot = the sum of the componentwise products of left and right
    │   │   └── return dot
    │   ├── function vectorCross(left, right) [local]
    │   │   ├── # Resolves the cross product of two [x, y, z] arrays.
    │   │   ├── impls cross = the right-handed cross product of left and right
    │   │   └── return cross
    │   ├── function vectorNormalize(vector) [local]
    │   │   ├── # Scales an [x, y, z] array to unit length.
    │   │   ├── calls vectorDot(vector, vector)
    │   │   ├── impls length = the square root of that squared length
    │   │   ├── if vector's length is zero or NaN  # a camera the polar band does not cover, or a NaN component, surfaced rather than handed on as a NaN pose
    │   │   │   └── throw cannot normalize a zero-length vector
    │   │   ├── impls unit = vector scaled by one over its length
    │   │   └── return unit
    │   ├── function recordToVector(record) [local]
    │   │   ├── # Converts a Plotly {x, y, z} camera record to an [x, y, z] array.
    │   │   ├── impls vector = [record.x, record.y, record.z]
    │   │   └── return vector
    │   ├── function vectorToRecord(vector) [local]
    │   │   ├── # Converts an [x, y, z] array to a Plotly {x, y, z} camera record.
    │   │   ├── impls record = { x: vector[0], y: vector[1], z: vector[2] }
    │   │   └── return record
    │   └── return rollLockCallback
    ├── for each matched input in window.dash_clientside.callback_context.inputs_list[0]  # one { id, property, value } per graph the pattern matches, a graph a Dash callback adds after the page loaded included
    │   ├── calls resolveGraphElementId(input.id)
    │   ├── impls worldAxis = JSON.parse(atob(input.id.lock_roll)), the unit-length axis create_dash_trackball_camera_controls normalized and base64-encoded
    │   ├── calls createRollLockCallback(graphElementId, worldAxis)
    │   └── calls rollLockCallback(input.value)
    └── return window.dash_clientside.no_update
```

`data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls.ts`

```text
trackball_camera_controls.ts
├── import * as THREE from "three";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45
│   └── # Shared vertical-FOV (degrees) every TS spatial display must construct its THREE.PerspectiveCamera with — 45° is the standard 50mm-equivalent lens FOV, trading perspective realism against off-center foreshortening for the orbit-around-near-scene-content use case this lib targets.
├── const ROLL_LOCKED_POLAR_ANGLE_EPSILON = 1e-6  # radians the roll-locked camera stops short of either pole of the lock axis
├── interface ThreeTrackballCameraControls
│   ├── getCameraState
│   │   └── # serializes the entire camera state (every CameraState field — both intrinsics and extrinsics) into a CameraState
│   ├── applyCameraState
│   │   └── # applies the entire CameraState (every field — both intrinsics and extrinsics) to the underlying camera and controls
│   ├── subscribeCameraStateChange
│   ├── target
│   ├── noRotate
│   ├── noZoom
│   ├── noPan
│   ├── minDistance
│   ├── maxDistance
│   ├── rollLockAxis               # the unit-length axis roll is locked about; null on the free trackball
│   ├── rollLockPolarAngleEpsilon  # ROLL_LOCKED_POLAR_ANGLE_EPSILON on a roll-locked construction; null on the free trackball
│   ├── addEventListener
│   ├── handleResize
│   └── update
├── function createTrackballCameraControls({ container, camera, renderer, initialCameraState, lockRoll = null })  # lockRoll: a non-zero THREE.Vector3 world-space axis of any length, or null for the free trackball
│   ├── # Builds, validates, and returns the trackball controls, seeding them from initialCameraState and observing the container's data-camera-state attribute for external sync.
│   ├── calls createRendererTrackballCameraControls({ camera, renderer, lockRoll })
│   ├── calls assertTrackballCameraControls({ controls, camera, renderer, lockRoll })
│   ├── if initialCameraState is not null
│   │   └── calls controls.applyCameraState(initialCameraState)
│   ├── impls MutationObserver on container's `data-camera-state` attribute → controls.applyCameraState(parsed state)
│   └── return
├── function createRendererTrackballCameraControls({ camera, renderer, lockRoll })
│   ├── # Constructs the renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu  # impls-node-one-step:skip
│   ├── if lockRoll is not null
│   │   ├── impls rollLockAxis = lockRoll normalized to unit length
│   │   ├── impls controls.rollLockAxis = rollLockAxis
│   │   ├── impls controls.rollLockPolarAngleEpsilon = ROLL_LOCKED_POLAR_ANGLE_EPSILON
│   │   ├── impls threeControls.noRotate = true, so the roll-locked left-drag below replaces three's free rotation while its right-drag pan and wheel zoom stay
│   │   ├── impls heldEyeOffset = a zero vector, the eye offset each hold below leaves behind for the next
│   │   ├── calls holdRollLockedCameraPose({ camera, target: threeControls.target, rollLockAxis, heldEyeOffset })  # the framing the controls are constructed on
│   │   ├── impls leftDrag = no left drag active, the drag state the handlers below share
│   │   ├── function startRollLockedLeftDrag(event) [local]  # event: a pointer press on renderer.domElement
│   │   │   ├── # Starts a roll-locked left drag at the pointer a left-button press lands on.
│   │   │   ├── if event is not a left-button press
│   │   │   │   └── return
│   │   │   └── impls leftDrag = active, from the event's pointer position
│   │   ├── impls renderer.domElement.addEventListener("pointerdown", startRollLockedLeftDrag)
│   │   ├── function endRollLockedLeftDrag() [local]
│   │   │   ├── # Ends the roll-locked left drag wherever the pointer is released.
│   │   │   └── impls leftDrag = no left drag active
│   │   ├── impls window.addEventListener("pointerup", endRollLockedLeftDrag)
│   │   ├── function turnRollLockedLeftDrag(event) [local]  # event: a pointer move anywhere on the page
│   │   │   ├── # Turns the camera by one left-drag pointer move, as yaw about rollLockAxis plus pitch about the camera right axis.
│   │   │   ├── if no left drag is active
│   │   │   │   └── return
│   │   │   ├── impls radiansPerPixel = threeControls.rotateSpeed / (0.5 × renderer.domElement.clientWidth), the free trackball's own rotation per pixel
│   │   │   ├── calls resolveRollLockBandedOffset({ offset: camera.position minus threeControls.target, rollLockAxis })   → bandedOffset
│   │   │   ├── impls yaws bandedOffset about rollLockAxis by minus the horizontal pointer delta from leftDrag times radiansPerPixel
│   │   │   ├── impls cameraRightAxis = normalize(cross(-bandedOffset, rollLockAxis))
│   │   │   ├── impls polarAngle = the angle from rollLockAxis to bandedOffset
│   │   │   ├── impls pitchAngle = minus the vertical pointer delta from leftDrag times radiansPerPixel
│   │   │   ├── impls pitchAngle = pitchAngle clamped so polarAngle + pitchAngle stays inside [ROLL_LOCKED_POLAR_ANGLE_EPSILON, π − ROLL_LOCKED_POLAR_ANGLE_EPSILON]  # a drag stops short of either pole
│   │   │   ├── impls pitches bandedOffset about cameraRightAxis by pitchAngle
│   │   │   ├── impls leftDrag = the event's pointer position, the one the next move's delta is measured from
│   │   │   ├── impls camera.position = threeControls.target + bandedOffset
│   │   │   ├── calls holdRollLockedCameraPose({ camera, target: threeControls.target, rollLockAxis, heldEyeOffset })
│   │   │   └── impls threeControls.dispatchEvent({ type: "change" })
│   │   ├── impls window.addEventListener("pointermove", turnRollLockedLeftDrag), so it runs for each left-drag pointer move
│   │   ├── impls freeApplyCameraState = controls.applyCameraState, the free trackball's own camera-state write kept for rollLockedApplyCameraState to apply through
│   │   ├── function rollLockedApplyCameraState(cameraState) [local]
│   │   │   ├── # Applies a camera state as the free trackball does, then re-holds the roll-locked pose it leaves.
│   │   │   ├── impls freeApplyCameraState(cameraState), applying it the way the free trackball applies it
│   │   │   └── calls holdRollLockedCameraPose({ camera, target: threeControls.target, rollLockAxis, heldEyeOffset })
│   │   ├── impls controls.applyCameraState = rollLockedApplyCameraState, so each state it is handed is re-held on the lock
│   │   ├── function rollLockedUpdate() [local]
│   │   │   ├── # Holds the roll-locked pose before three's own update, so a target or position a caller writes directly is held from the next update on.
│   │   │   ├── calls holdRollLockedCameraPose({ camera, target: threeControls.target, rollLockAxis, heldEyeOffset })
│   │   │   └── impls three's own trackball update of threeControls
│   │   ├── impls threeControls.update = rollLockedUpdate
│   │   └── return controls
│   └── return controls  # the free trackball controls exactly as three constructed them, so a caller naming no axis renders what it rendered before this argument existed
├── function holdRollLockedCameraPose({ camera, target, rollLockAxis, heldEyeOffset })  # heldEyeOffset: the banded eye offset the previous hold left behind
│   ├── # Holds the camera on the roll-locked pose its own framing implies, the eye banded off the lock axis and camera.up re-derived from the view direction and that axis.
│   ├── impls offset = camera.position minus target
│   ├── if offset has zero length  # an eye written onto the target, or the target onto the eye, names no view direction
│   │   └── impls offset = heldEyeOffset
│   ├── calls resolveRollLockBandedOffset({ offset, rollLockAxis })
│   ├── impls heldEyeOffset = bandedOffset, overwritten in place for the next hold
│   ├── impls cameraRightAxis = normalize(cross(-bandedOffset, rollLockAxis))
│   ├── impls camera.position = target + bandedOffset
│   ├── impls camera.up = normalize(cross(cameraRightAxis, normalize(-bandedOffset)))
│   ├── impls camera.lookAt(target)
│   └── return
├── function resolveRollLockBandedOffset({ offset, rollLockAxis }: { offset: THREE.Vector3; rollLockAxis: THREE.Vector3 }): THREE.Vector3
│   ├── # Bands an eye offset's polar angle off the lock axis into [ROLL_LOCKED_POLAR_ANGLE_EPSILON, π − ROLL_LOCKED_POLAR_ANGLE_EPSILON], rebuilding it at the banded angle on its own meridian.
│   ├── if the offset's polar angle already lies inside the band
│   │   └── return offset
│   ├── calls resolveRollLockMeridian({ offset, rollLockAxis })   → meridian
│   ├── impls bandedPolarAngle = the offset's polar angle clamped into [ROLL_LOCKED_POLAR_ANGLE_EPSILON, π − ROLL_LOCKED_POLAR_ANGLE_EPSILON]
│   ├── impls radius = the length of offset
│   ├── impls bandedOffset = meridian × radius·sin(bandedPolarAngle) + rollLockAxis × radius·cos(bandedPolarAngle)
│   └── return bandedOffset
├── function resolveRollLockMeridian({ offset, rollLockAxis }: { offset: THREE.Vector3; rollLockAxis: THREE.Vector3 }): THREE.Vector3
│   ├── # Resolves the meridian an eye offset stands on, as a unit vector perpendicular to the lock axis.
│   ├── impls meridian = offset minus its projection onto rollLockAxis
│   ├── if meridian has non-zero length
│   │   ├── impls meridian.normalize()
│   │   └── return meridian
│   ├── impls axisMagnitudes = the absolute value of each component of rollLockAxis  # an offset on the axis stands on every meridian at once
│   ├── impls leastLeanedBasisVector = the world basis vector along the smallest of axisMagnitudes, the one rollLockAxis leans on least
│   ├── impls fallbackMeridian = normalize(cross(rollLockAxis, leastLeanedBasisVector))
│   └── return fallbackMeridian
├── function assertTrackballCameraControls({ controls, camera, renderer, lockRoll }: { controls: ThreeTrackballCameraControls; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; lockRoll: THREE.Vector3 | null }): void
│   ├── # Validates the constructed controls satisfy every trackball contract by running the mouse-mapping, no-orbit, no-pose-clamp, and roll-lock assertions.
│   ├── calls assertTrackballMouseMapping({ controls, renderer })
│   ├── calls assertNoOrbitCameraControls({ controls })
│   ├── calls assertNoCameraPoseClamps({ controls, lockRoll })
│   ├── calls assertRollLock({ controls, camera, lockRoll })
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
│   ├── if controls are not three's TrackballControls  # orbit-style target-locked controls are what this rules out
│   │   └── throw orbit-style camera controls are forbidden
│   └── return
├── function assertNoCameraPoseClamps({ controls, lockRoll }: { controls: ThreeTrackballCameraControls; lockRoll: THREE.Vector3 | null }): void
│   ├── # Asserts the controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation beyond the polar band a roll lock costs.
│   ├── if controls disable pan or bound the eye distance  # the azimuth, target-lock, distance and translation restrictions three's TrackballControls can carry
│   │   └── throw restricted camera pose controls
│   ├── if lockRoll is null and controls restrict polar angle or rotation
│   │   └── throw restricted camera pose controls
│   ├── if lockRoll is not null and three's rotation is off with no roll-locked rotation replacing it
│   │   └── throw roll lock must cost only the roll axis and the polar extremes
│   └── return
└── function assertRollLock({ controls, camera, lockRoll }: { controls: ThreeTrackballCameraControls; camera: THREE.PerspectiveCamera; lockRoll: THREE.Vector3 | null }): void
    ├── # Asserts roll is held about lockRoll when one is supplied and left free when none is, this module owning no axis of its own.
    ├── if lockRoll is not null and controls do not hold rollLockAxis as lockRoll normalized  # the axis the roll-locked drag keeps the camera right axis perpendicular to
    │   └── throw roll-locked camera controls must keep the camera right axis perpendicular to the supplied axis
    ├── if lockRoll is not null and controls let the camera up vector cross to the far side of lockRoll
    │   └── throw roll-locked camera controls must keep the camera up vector on the supplied axis's side
    ├── if lockRoll is null and controls hold a rollLockAxis or a rollLockPolarAngleEpsilon  # either constrains the camera right axis against an axis
    │   └── throw free trackball camera controls must leave camera roll unconstrained
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
