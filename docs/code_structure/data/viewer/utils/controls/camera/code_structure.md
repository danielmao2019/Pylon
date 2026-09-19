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
├── import * as THREE from "three";
├── import { TrackballControls as ThreeTrackballControlsImpl } from "three/examples/jsm/controls/TrackballControls.js";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45
│   └── # Shared vertical-FOV (degrees) every TS spatial display must construct its THREE.PerspectiveCamera with — 45° is the standard 50mm-equivalent lens FOV, trading perspective realism against off-center foreshortening for the orbit-around-near-scene-content use case this lib targets.
├── type CameraStateListener = (cameraState: CameraState) => void
├── interface RendererTrackballCameraControls
│   ├── targetElement: HTMLElement
│   ├── getCameraState: () => CameraState | null
│   ├── applyCameraState: (cameraState: CameraState | null) => void
│   └── subscribeCameraStateChange: (listener: CameraStateListener) => () => void
├── export interface TrackballCameraControls
│   ├── getCameraState: () => CameraState | null
│   ├── applyCameraState: (cameraState: CameraState | null) => void
│   └── subscribeCameraStateChange: (listener: CameraStateListener) => () => void
├── export interface ThreeTrackballCameraControls extends TrackballCameraControls
│   ├── target: THREE.Vector3
│   ├── addEventListener: (type: "change", listener: () => void) => void
│   ├── handleResize: () => void
│   └── update: () => void
├── export function createTrackballCameraControls(args: { targetElement: HTMLElement; initialCameraState?: CameraState | null; }): TrackballCameraControls;
│   └── # Declares the element form: trackball controls over a target element.
├── export function createTrackballCameraControls(args: { camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; container: HTMLElement; initialCameraState?: CameraState | null; }): ThreeTrackballCameraControls;
│   └── # Declares the three.js form: trackball controls over a camera, its renderer and its container.
├── export function createTrackballCameraControls(args: | { targetElement: HTMLElement; initialCameraState?: CameraState | null; } | { camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; container: HTMLElement; initialCameraState?: CameraState | null; }, ): TrackballCameraControls | ThreeTrackballCameraControls
│   ├── # Builds the trackball controls in the form its args select: three.js controls for a camera, else validated controls over the target element.
│   ├── if ("camera" in args)
│   │   ├── calls createThreeTrackballCameraControls(args)
│   │   └── return
│   ├── impls const { targetElement, initialCameraState = null } = args
│   ├── calls createRendererTrackballCameraControls({ targetElement, initialCameraState, })  # -> controls
│   ├── calls assertTrackballCameraControls(controls)
│   └── return controls
├── function createThreeTrackballCameraControls(args: { camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; container: HTMLElement; initialCameraState?: CameraState | null; }): ThreeTrackballCameraControls
│   ├── # Wraps three.js TrackballControls on the renderer's canvas as trackball camera controls, seeded from initialCameraState and re-applying the container's data-camera-state attribute on every change.
│   ├── impls const { camera, renderer, container, initialCameraState = null } = args
│   ├── impls const threeControls = new ThreeTrackballControlsImpl(camera, renderer.domElement)
│   ├── impls const listeners = new Set<CameraStateListener>()
│   ├── impls threeControls.rotateSpeed = 3
│   ├── impls threeControls.zoomSpeed = 1.5
│   ├── impls threeControls.panSpeed = 0.8
│   ├── impls threeControls.staticMoving = true
│   ├── (event: MouseEvent) => [local]
│   │   ├── # The canvas contextmenu listener: suppresses the browser menu so right-drag pans.
│   │   └── impls event.preventDefault()
│   ├── impls renderer.domElement.addEventListener("contextmenu", that listener)
│   ├── () => [local]
│   │   ├── # The controls change listener: serializes the camera and hands the state to every subscribed listener.
│   │   ├── calls buildThreeTrackballCameraState({ camera, controls: threeControls, })  # -> cameraState
│   │   └── for (const listener of listeners)
│   │       └── calls listener(cameraState)
│   ├── impls threeControls.addEventListener("change", that listener)
│   ├── () => [local]
│   │   ├── # getCameraState: serializes the camera and its controls.
│   │   └── calls buildThreeTrackballCameraState({ camera, controls: threeControls, })
│   ├── (cameraState: CameraState | null): void => [local]
│   │   ├── # applyCameraState: applies the state onto the camera and its controls.
│   │   └── calls applyThreeTrackballCameraState({ camera, controls: threeControls, cameraState, })
│   ├── (listener: CameraStateListener) => [local]
│   │   ├── # subscribeCameraStateChange: registers the listener and returns its unsubscribe.
│   │   ├── if (typeof listener !== "function")
│   │   │   └── throw new Error("camera state listener must be a function")
│   │   ├── impls listeners.add(listener)
│   │   ├── () => [local]
│   │   │   ├── # The unsubscribe: removes the listener.
│   │   │   └── impls listeners.delete(listener)
│   │   └── return
│   ├── impls const result = Object.assign(threeControls, { getCameraState, applyCameraState, subscribeCameraStateChange }), each being its lambda above
│   ├── if (initialCameraState !== null)
│   │   └── calls result.applyCameraState(initialCameraState)
│   ├── () => [local]
│   │   ├── # The container observer: re-applies the container's data-camera-state attribute, skipping an unparseable value.
│   │   ├── impls const raw = container.dataset.cameraState
│   │   ├── if (raw === undefined)
│   │   │   └── return
│   │   ├── try
│   │   │   └── calls result.applyCameraState(JSON.parse(raw) as CameraState)
│   │   └── catch
│   │       └── # ignore unparseable dataset values
│   ├── impls const observer = new MutationObserver(that observer callback)
│   ├── impls observer.observe(container, { attributeFilter: ["data-camera-state"], attributes: true, })
│   └── return result
├── function createRendererTrackballCameraControls(args: { targetElement: HTMLElement; initialCameraState: CameraState | null; }): RendererTrackballCameraControls
│   ├── # Builds trackball controls over a target element, syncing its camera state across the element's data-camera-state attribute, its embedded renderer iframe and its listeners, and marking the element with the trackball contract.
│   ├── impls const { targetElement, initialCameraState } = args
│   ├── impls let currentCameraState = initialCameraState
│   ├── impls let internallyWrittenCameraStateToken: string | null | undefined = undefined
│   ├── impls const listeners: CameraStateListener[] = []
│   ├── const setInternallyWrittenCameraStateToken = ( token: string | null, ): void => [local]
│   │   ├── # Records the token these controls last wrote onto the element, so the observer skips their own write.
│   │   └── impls internallyWrittenCameraStateToken = token
│   ├── const applyCameraState = (cameraState: CameraState | null): void => [local]
│   │   ├── # Applies a caller-given camera state: keeps it, writes it onto the element, and posts it to the embedded renderer.
│   │   ├── impls currentCameraState = cameraState
│   │   ├── calls writeInternalCameraStateToTargetElement({ targetElement, cameraState, setInternallyWrittenCameraStateToken, })
│   │   └── calls postCameraStateToEmbeddedRenderer({ targetElement, cameraState, })
│   ├── const emitCameraStateChange = (cameraState: CameraState): void => [local]
│   │   ├── # Publishes a renderer-reported camera state: keeps it, writes it onto the element, hands it to every listener, and dispatches a bubbling camera-pose-change event.
│   │   ├── impls currentCameraState = cameraState
│   │   ├── calls writeInternalCameraStateToTargetElement({ targetElement, cameraState, setInternallyWrittenCameraStateToken, })
│   │   ├── for (const listener of listeners)
│   │   │   └── calls listener(cameraState)
│   │   └── impls targetElement.dispatchEvent(new CustomEvent<CameraState>("camera-pose-change", { bubbles: true, detail: cameraState, }))
│   ├── () => [local]
│   │   ├── # The element observer: skips the controls' own write, and adopts any other write of the element's camera state.
│   │   ├── calls readCameraStateTokenFromTargetElement(targetElement)  # -> targetElementCameraStateToken
│   │   ├── if ( internallyWrittenCameraStateToken !== undefined && targetElementCameraStateToken === internallyWrittenCameraStateToken )
│   │   │   ├── impls internallyWrittenCameraStateToken = undefined
│   │   │   └── return
│   │   ├── impls internallyWrittenCameraStateToken = undefined
│   │   ├── calls readCameraStateFromTargetElement(targetElement)
│   │   └── calls applyExternalCameraState(readCameraStateFromTargetElement(targetElement))
│   ├── impls const mutationObserver = new MutationObserver(that observer callback)
│   ├── impls mutationObserver.observe(targetElement, { attributeFilter: ["data-camera-state"], attributes: true, })
│   ├── (event: MessageEvent<unknown>) => [local]
│   │   ├── # The window message listener: emits the camera state that the element's own embedded renderer reports from this origin.
│   │   ├── calls isEmbeddedRendererMessageSource({ targetElement, source: event.source })
│   │   ├── if (!isEmbeddedRendererMessageSource({ targetElement, source: event.source }))
│   │   │   └── return
│   │   ├── if (event.origin !== window.location.origin)
│   │   │   └── return
│   │   ├── impls const message = event.data
│   │   ├── calls isTrackballCameraStateChangeMessage(message)
│   │   ├── if (!isTrackballCameraStateChangeMessage(message))
│   │   │   └── return
│   │   └── calls emitCameraStateChange(message.cameraState)
│   ├── impls window.addEventListener("message", that listener)
│   ├── if (targetElement instanceof HTMLIFrameElement)
│   │   ├── () => [local]
│   │   │   ├── # The iframe load listener: posts the kept camera state to the freshly loaded renderer.
│   │   │   └── calls postCameraStateToEmbeddedRenderer({ targetElement, cameraState: currentCameraState, })
│   │   └── impls targetElement.addEventListener("load", that listener)
│   ├── impls targetElement.dataset.cameraControlMode = "trackball"
│   ├── impls targetElement.dataset.trackballMouseMapping = "left-drag-rotate/right-drag-pan/wheel-zoom"
│   ├── impls targetElement.dataset.contextMenuBehavior = "suppressed-for-trackball-pan"
│   ├── if (currentCameraState !== null)
│   │   └── calls applyCameraState(currentCameraState)
│   ├── function applyExternalCameraState(cameraState: CameraState | null): void [local]
│   │   ├── # Adopts a camera state written onto the element from outside: keeps it and posts it to the embedded renderer.
│   │   ├── impls currentCameraState = cameraState
│   │   └── calls postCameraStateToEmbeddedRenderer({ targetElement, cameraState, })
│   ├── () => [local]
│   │   ├── # getCameraState: reads the kept camera state.
│   │   └── impls currentCameraState
│   ├── (listener: CameraStateListener) => [local]
│   │   ├── # subscribeCameraStateChange: registers the listener and returns its unsubscribe.
│   │   ├── if (typeof listener !== "function")
│   │   │   └── throw new Error("camera state listener must be a function")
│   │   ├── impls listeners.push(listener)
│   │   ├── () => [local]
│   │   │   ├── # The unsubscribe: removes the listener while it is still registered.
│   │   │   ├── impls const index = listeners.indexOf(listener)
│   │   │   └── if (index >= 0)
│   │   │       └── impls listeners.splice(index, 1)
│   │   └── return
│   └── return { targetElement, getCameraState, applyCameraState, subscribeCameraStateChange }, getCameraState and subscribeCameraStateChange being their lambdas above
├── function assertTrackballCameraControls( controls: RendererTrackballCameraControls, ): void
│   ├── # Validates the constructed controls satisfy every trackball contract by running the mouse-mapping, no-orbit, and no-pose-clamp assertions.
│   ├── calls assertTrackballMouseMapping(controls)
│   ├── calls assertNoOrbitCameraControls(controls)
│   └── calls assertNoCameraPoseClamps(controls)
├── function assertTrackballMouseMapping( controls: RendererTrackballCameraControls, ): void
│   ├── # Asserts the target element is marked with the left-drag-rotate / right-drag-pan / wheel-zoom mapping and with context-menu suppression for trackball panning.
│   ├── impls const mapping = controls.targetElement.dataset.trackballMouseMapping
│   ├── if (mapping !== "left-drag-rotate/right-drag-pan/wheel-zoom")
│   │   └── throw new Error("invalid trackball camera controls")
│   └── if ( controls.targetElement.dataset.contextMenuBehavior !== "suppressed-for-trackball-pan" )
│       └── throw new Error("context menu blocks trackball panning")
├── function assertNoOrbitCameraControls( controls: RendererTrackballCameraControls, ): void
│   ├── # Asserts the target element is marked with neither an orbit camera-control mode nor an orbit camera-control family.
│   ├── impls const mode = controls.targetElement.dataset.cameraControlMode
│   ├── impls const family = controls.targetElement.dataset.cameraControlFamily
│   └── if (mode === "orbit" || family === "orbit")
│       └── throw new Error("orbit-style camera controls are forbidden")
├── function assertNoCameraPoseClamps( controls: RendererTrackballCameraControls, ): void
│   ├── # Asserts the target element carries none of the camera-pose restriction keys on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
│   ├── impls const forbiddenRestrictionKeys = [ "cameraPolarAngleLimit", "cameraAzimuthAngleLimit", "cameraTargetLock", "cameraDistanceBounds", "cameraPanLimit", "cameraTranslationLimit", "cameraRotationLimit", ]
│   ├── (key) => [local]
│   │   ├── # The find predicate: whether the target element carries that restriction key.
│   │   └── impls controls.targetElement.dataset[key] !== undefined
│   ├── impls const restrictedKey = forbiddenRestrictionKeys.find(that predicate)
│   └── if (restrictedKey !== undefined)
│       └── throw new Error(`restricted camera pose controls: ${restrictedKey}`)
├── function buildThreeTrackballCameraState({ camera, controls, }: { camera: THREE.PerspectiveCamera; controls: ThreeTrackballControlsImpl; }): CameraState
│   ├── # Serializes a three.js perspective camera and its trackball controls into a CameraState.
│   ├── calls vectorToRecord(camera.position)
│   ├── calls quaternionToRecord(camera.quaternion)
│   ├── calls vectorToRecord(controls.target)
│   ├── calls vectorToRecord(camera.up)
│   └── return  # the CameraState: perspective-three intrinsics off the camera, the converted position, quaternion, target and up, three_trackball conventions, null name and id
├── function applyThreeTrackballCameraState({ camera, controls, cameraState, }: { camera: THREE.PerspectiveCamera; controls: ThreeTrackballControlsImpl; cameraState: CameraState | null; }): void
│   ├── # Applies a three_trackball CameraState back onto a three.js perspective camera and its trackball controls.
│   ├── if cameraState is null or its extr_convention is not "three_trackball"
│   │   └── return
│   ├── impls position = cameraState.extrinsics.position
│   ├── impls quaternion = cameraState.extrinsics.quaternion
│   ├── impls target = cameraState.extrinsics.target
│   ├── impls up = cameraState.extrinsics.up
│   ├── impls aspect = cameraState.intrinsics.aspect
│   ├── impls far = cameraState.intrinsics.far
│   ├── impls fov = cameraState.intrinsics.fov
│   ├── impls near = cameraState.intrinsics.near
│   ├── calls isVectorRecord(position)
│   ├── calls isQuaternionRecord(quaternion)
│   ├── calls isVectorRecord(target)
│   ├── calls isVectorRecord(up)
│   ├── if any of position, target and up is not a vector record, quaternion is not a quaternion record, or any of aspect, far, fov and near is not a number
│   │   └── return
│   ├── impls camera.position.set(position.x, position.y, position.z)
│   ├── impls camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
│   ├── impls camera.up.set(up.x, up.y, up.z)
│   ├── impls controls.target.set(target.x, target.y, target.z)
│   ├── impls camera.aspect = aspect
│   ├── impls camera.far = far
│   ├── impls camera.fov = fov
│   ├── impls camera.near = near
│   ├── impls camera.updateProjectionMatrix()
│   └── impls controls.update()
├── function vectorToRecord(vector: THREE.Vector3): Record<string, number>
│   ├── # Converts a three.js vector into the plain x / y / z record a CameraState carries.
│   └── return  # the record holding the vector's x, y and z
├── function quaternionToRecord(quaternion: THREE.Quaternion, ): Record<string, number>
│   ├── # Converts a three.js quaternion into the plain x / y / z / w record a CameraState carries.
│   └── return  # the record holding the quaternion's x, y, z and w
├── function isQuaternionRecord(value: unknown): value is { x: number; y: number; z: number; w: number; }
│   ├── # Narrows an unknown CameraState field to a quaternion record.
│   ├── calls isVectorRecord(value)
│   └── return  # whether value is a vector record whose w is a number too
├── function isVectorRecord(value: unknown): value is { x: number; y: number; z: number; }
│   ├── # Narrows an unknown CameraState field to a vector record.
│   └── return  # whether value is a non-null object whose x, y and z are all numbers
├── function writeInternalCameraStateToTargetElement(args: { targetElement: HTMLElement; cameraState: CameraState | null; setInternallyWrittenCameraStateToken: (token: string | null) => void; }): void
│   ├── # Writes a camera state onto the element's data-camera-state attribute and, when that changed the attribute, records its token as the controls' own write.
│   ├── impls const { targetElement, cameraState, setInternallyWrittenCameraStateToken, } = args
│   ├── calls serializeCameraState(cameraState)  # -> serializedCameraState
│   ├── calls writeCameraStateToTargetElement({ targetElement, cameraState, serializedCameraState, })
│   └── if ( writeCameraStateToTargetElement({ targetElement, cameraState, serializedCameraState, }) )
│       └── calls setInternallyWrittenCameraStateToken(serializedCameraState)
├── function serializeCameraState(cameraState: CameraState | null): string | null
│   ├── # Serializes a camera state into its JSON token, a null state staying null.
│   ├── if (cameraState === null)
│   │   └── return null
│   └── return JSON.stringify(cameraState)
├── function writeCameraStateToTargetElement(args: { targetElement: HTMLElement; cameraState: CameraState | null; serializedCameraState: string | null; }): boolean
│   ├── # Writes the serialized camera state onto the element's data-camera-state attribute, deleting the attribute for a null state, and returns whether the attribute changed.
│   ├── impls const { targetElement, cameraState, serializedCameraState } = args
│   ├── calls readCameraStateTokenFromTargetElement(targetElement)
│   ├── if ( readCameraStateTokenFromTargetElement(targetElement) === serializedCameraState )
│   │   └── return false
│   ├── if (cameraState === null)
│   │   ├── impls delete targetElement.dataset.cameraState
│   │   └── return true
│   ├── if (serializedCameraState === null)
│   │   └── throw new Error("serialized camera state is unexpectedly null")
│   ├── impls targetElement.dataset.cameraState = serializedCameraState
│   └── return true
├── function readCameraStateTokenFromTargetElement( targetElement: HTMLElement, ): string | null
│   ├── # Reads the element's raw data-camera-state token, null when the attribute is absent.
│   └── return targetElement.dataset.cameraState ?? null
├── function postCameraStateToEmbeddedRenderer(args: { targetElement: HTMLElement; cameraState: CameraState | null; }): void
│   ├── # Posts a camera state to the element's embedded renderer window, when the element is an iframe holding one.
│   ├── impls const { targetElement, cameraState } = args
│   ├── if (!(targetElement instanceof HTMLIFrameElement))
│   │   └── return
│   ├── impls const targetWindow = targetElement.contentWindow
│   ├── if (targetWindow === null)
│   │   └── return
│   └── impls targetWindow.postMessage({ cameraState, type: "trackball-camera-state", }, window.location.origin)
├── function readCameraStateFromTargetElement( targetElement: HTMLElement, ): CameraState | null
│   ├── # Parses the element's data-camera-state attribute into a validated CameraState, null when the attribute is absent.
│   ├── impls const serializedCameraState = targetElement.dataset.cameraState
│   ├── if (serializedCameraState === undefined)
│   │   └── return null
│   ├── impls const parsedCameraState: unknown = JSON.parse(serializedCameraState)
│   ├── calls isCameraState(parsedCameraState)
│   ├── if (!isCameraState(parsedCameraState))
│   │   └── throw new Error("target camera state does not match CameraState")
│   └── return parsedCameraState
├── function isEmbeddedRendererMessageSource(args: { targetElement: HTMLElement; source: MessageEventSource | null; }): boolean
│   ├── # Checks whether a message came from the window of the element's own embedded renderer iframe.
│   ├── impls const { targetElement, source } = args
│   └── return (targetElement instanceof HTMLIFrameElement && source !== null && source === targetElement.contentWindow)
├── function isTrackballCameraStateChangeMessage( value: unknown, ): value is { type: "trackball-camera-state-change"; cameraState: CameraState }
│   ├── # Narrows an unknown message to a trackball camera-state-change message carrying a CameraState.
│   ├── calls isRecord(value)
│   ├── calls isCameraState(value.cameraState)
│   └── return  # whether value is a record typed "trackball-camera-state-change" whose cameraState is a CameraState
├── function isCameraState(value: unknown): value is CameraState
│   ├── # Narrows an unknown value to a CameraState by its record fields, string conventions, and null-or-string name and id.
│   ├── calls isRecord(value)
│   ├── calls isRecord(value.intrinsics)
│   ├── calls isRecord(value.extrinsics)
│   └── return  # whether value is a record whose intrinsics and extrinsics are records, whose intr_convention and extr_convention are strings, and whose name and id are each null or a string
└── function isRecord(value: unknown): value is Record<string, unknown>
    ├── # Narrows an unknown value to a non-null object record.
    └── return value !== null && typeof value === "object"
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
