import * as THREE from "three";
import { TrackballControls as ThreeTrackballControlsImpl } from "three/examples/jsm/controls/TrackballControls.js";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";

export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45;

// Radians the roll-locked camera stops short of either pole of the lock axis. The incoming offset is banded into this range before anything derives a camera right axis from it, and the pitch is then clamped to keep it there, so the view direction never runs parallel to the axis and the cross product that re-derives the camera right axis never collapses.
const ROLL_LOCKED_POLAR_ANGLE_EPSILON = 1e-6;

export interface ThreeTrackballCameraControls {
  getCameraState: () => CameraState | null;
  applyCameraState: (cameraState: CameraState | null) => void;
  subscribeCameraStateChange: (
    listener: (cameraState: CameraState) => void,
  ) => () => void;
  target: THREE.Vector3;
  noRotate: boolean;
  noZoom: boolean;
  noPan: boolean;
  minDistance: number;
  maxDistance: number;
  rollLockAxis: THREE.Vector3 | null;
  rollLockPolarAngleEpsilon: number | null;
  addEventListener: (type: "change", listener: () => void) => void;
  handleResize: () => void;
  update: () => void;
}

// Builds, validates, and returns the trackball controls, seeding them from initialCameraState and observing the container's data-camera-state attribute for external sync.
//
// Args:
//   container: the display container the controls observe data-camera-state on.
//   camera: the perspective camera the controls drive.
//   renderer: the WebGL renderer whose canvas receives the pointer events.
//   initialCameraState: initial framing (camera-to-world extrinsics + intrinsics); null uses the camera's default framing.
//   lockRoll: a non-zero world-space axis of any length to lock camera roll about, in the scene's own world frame; null leaves the controls as the free trackball, with no roll-locked rotation installed. This module owns no axis of its own, so the axis is always the caller's.
//
// Returns:
//   The validated trackball controls.
export function createTrackballCameraControls({
  container,
  camera,
  renderer,
  initialCameraState = null,
  lockRoll = null
}: {
  container: HTMLElement;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  initialCameraState?: CameraState | null;
  lockRoll?: THREE.Vector3 | null
}): ThreeTrackballCameraControls {
  const _validate_inputs = (): void => {
    if (lockRoll !== null) {
      if (!(lockRoll instanceof THREE.Vector3)) {
        throw new Error(
          `lockRoll must be a THREE.Vector3 or null: lockRoll=${JSON.stringify(lockRoll)}`,
        );
      }
      if (
        !Number.isFinite(lockRoll.x) ||
        !Number.isFinite(lockRoll.y) ||
        !Number.isFinite(lockRoll.z)
      ) {
        throw new Error(
          `lockRoll must have finite components: lockRoll=(${lockRoll.x}, ${lockRoll.y}, ${lockRoll.z})`,
        );
      }
      if (lockRoll.lengthSq() === 0) {
        throw new Error(
          `lockRoll must have non-zero length: lockRoll=(${lockRoll.x}, ${lockRoll.y}, ${lockRoll.z})`,
        );
      }
    }
  };
  _validate_inputs();

  const controls = createRendererTrackballCameraControls({
    camera,
    renderer,
    lockRoll,
  });
  assertTrackballCameraControls({ controls, camera, renderer, lockRoll });
  if (initialCameraState !== null) {
    controls.applyCameraState(initialCameraState);
  }
  const observer = new MutationObserver(() => {
    const serializedCameraState = container.dataset.cameraState;
    if (serializedCameraState === undefined) {
      return;
    }
    controls.applyCameraState(JSON.parse(serializedCameraState) as CameraState);
  });
  observer.observe(container, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });
  return controls;
}

// Constructs the renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
//
// Args:
//   camera: the perspective camera the controls drive.
//   renderer: the WebGL renderer whose canvas receives the pointer events.
//   lockRoll: the world-space axis to lock camera roll about; null returns the free trackball controls, with no roll-locked rotation installed, so a caller naming no axis renders what it rendered before this argument existed.
//
// Returns:
//   The renderer-specific trackball controls.
function createRendererTrackballCameraControls({
  camera,
  renderer,
  lockRoll
}: {
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  lockRoll: THREE.Vector3 | null
}): ThreeTrackballCameraControls {
  const threeControls = new ThreeTrackballControlsImpl(camera, renderer.domElement);
  const listeners = new Set<(cameraState: CameraState) => void>();
  threeControls.rotateSpeed = 3;
  threeControls.zoomSpeed = 1.5;
  threeControls.panSpeed = 0.8;
  threeControls.staticMoving = true;
  renderer.domElement.addEventListener("contextmenu", (event: MouseEvent) => {
    event.preventDefault();
  });
  threeControls.addEventListener("change", () => {
    const cameraState = buildThreeTrackballCameraState({
      camera,
      controls: threeControls,
    });
    for (const listener of listeners) {
      listener(cameraState);
    }
  });
  const controls: ThreeTrackballCameraControls = Object.assign(threeControls, {
    rollLockAxis: null,
    rollLockPolarAngleEpsilon: null,
    getCameraState: () =>
      buildThreeTrackballCameraState({
        camera,
        controls: threeControls,
      }),
    applyCameraState: (cameraState: CameraState | null): void => {
      applyThreeTrackballCameraState({
        camera,
        controls: threeControls,
        cameraState,
      });
    },
    subscribeCameraStateChange: (listener: (cameraState: CameraState) => void) => {
      if (typeof listener !== "function") {
        throw new Error("camera state listener must be a function");
      }
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  });

  if (lockRoll !== null) {
    const rollLockAxis = lockRoll.clone().normalize();
    controls.rollLockAxis = rollLockAxis;
    controls.rollLockPolarAngleEpsilon = ROLL_LOCKED_POLAR_ANGLE_EPSILON;

    // Three's own rotation is the free trackball that carries camera.up along with the drag; the roll-locked left-drag below replaces it, leaving three's right-drag pan and wheel zoom untouched.
    threeControls.noRotate = true;

    // The eye offset every hold below last left behind, held again in place of an eye written exactly onto the target, the one framing that names no offset of its own.
    const heldEyeOffset = new THREE.Vector3();

    // The framing the controls are constructed on is a pose like any other, and a caller is free to hand over one looking straight down the lock axis. Holding it here is what leaves the camera frame real before anything draws with it, rather than only once a drag has repaired it.
    holdRollLockedCameraPose({
      camera,
      target: threeControls.target,
      rollLockAxis,
      heldEyeOffset,
    });

    let leftDrag: { clientX: number; clientY: number } | null = null;

    // Starts a roll-locked left drag at the pointer a left-button press lands on.
    //
    // Args:
    //   event: a pointer press on renderer.domElement.
    //
    // Returns:
    //   void.
    function startRollLockedLeftDrag(event: PointerEvent): void {
      if (event.button !== 0) {
        return;
      }
      leftDrag = { clientX: event.clientX, clientY: event.clientY };
    }
    renderer.domElement.addEventListener("pointerdown", startRollLockedLeftDrag);

    // Ends the roll-locked left drag wherever the pointer is released.
    //
    // Args:
    //   None.
    //
    // Returns:
    //   void.
    function endRollLockedLeftDrag(): void {
      leftDrag = null;
    }
    window.addEventListener("pointerup", endRollLockedLeftDrag);

    // Turns the camera by one left-drag pointer move, as yaw about rollLockAxis plus pitch about the camera right axis.
    //
    // Args:
    //   event: a pointer move anywhere on the page.
    //
    // Returns:
    //   void.
    function turnRollLockedLeftDrag(event: PointerEvent): void {
      if (leftDrag === null) {
        return;
      }
      // Three's trackball turns by rotateSpeed per half canvas width of pointer travel, so the locked drag turns exactly as far per pixel as the free one.
      const radiansPerPixel =
        threeControls.rotateSpeed / (0.5 * renderer.domElement.clientWidth);
      // A pan moves the rotation target out from under the eye, so the pose a drag starts from can sit on the lock axis however the previous one was held. Banding it first is what leaves the camera right axis below derivable at all: on the axis that cross product collapses, three normalizes the collapse to a zero vector rather than a NaN one, so the pitch quaternion the clamp feeds is built about nothing and the camera never leaves the pole again.
      const bandedOffset = resolveRollLockBandedOffset({
        offset: camera.position.clone().sub(threeControls.target),
        rollLockAxis,
      });
      bandedOffset.applyAxisAngle(
        rollLockAxis,
        -(event.clientX - leftDrag.clientX) * radiansPerPixel,
      );
      const cameraRightAxis = new THREE.Vector3()
        .crossVectors(bandedOffset.clone().negate(), rollLockAxis)
        .normalize();
      // Yaw turns about the lock axis and so leaves the angle to it alone, which makes the pitch the whole of what can reach a pole. Clamping it to the band keeps the camera short of the pole, where a further pitch step toward it is rejected instead of carrying the view through and inverting the scene.
      const polarAngle = bandedOffset.angleTo(rollLockAxis);
      let pitchAngle = -(event.clientY - leftDrag.clientY) * radiansPerPixel;
      pitchAngle = Math.min(
        Math.max(pitchAngle, ROLL_LOCKED_POLAR_ANGLE_EPSILON - polarAngle),
        Math.PI - ROLL_LOCKED_POLAR_ANGLE_EPSILON - polarAngle,
      );
      bandedOffset.applyAxisAngle(cameraRightAxis, pitchAngle);
      leftDrag = { clientX: event.clientX, clientY: event.clientY };
      camera.position.addVectors(threeControls.target, bandedOffset);
      holdRollLockedCameraPose({
        camera,
        target: threeControls.target,
        rollLockAxis,
        heldEyeOffset,
      });
      threeControls.dispatchEvent({ type: "change" });
    }
    window.addEventListener("pointermove", turnRollLockedLeftDrag);

    // The free trackball's own camera-state write, kept for rollLockedApplyCameraState to apply through.
    const freeApplyCameraState = controls.applyCameraState;

    // Applies a camera state as the free trackball does, then re-holds the roll-locked pose it leaves.
    //
    // Args:
    //   cameraState: the camera state (camera-to-world extrinsics + intrinsics) to apply; null, or a state in another convention, leaves the camera as it was before the hold.
    //
    // Returns:
    //   void.
    function rollLockedApplyCameraState(cameraState: CameraState | null): void {
      freeApplyCameraState(cameraState);
      holdRollLockedCameraPose({
        camera,
        target: threeControls.target,
        rollLockAxis,
        heldEyeOffset,
      });
    }
    // A camera-sync peer, a restored framing, or the container's data-camera-state can hand over any pose at all, so each one is held on the lock the moment it lands.
    controls.applyCameraState = rollLockedApplyCameraState;

    // Holds the roll-locked pose before three's own update, so a target or position a caller writes directly is held from the next update on.
    //
    // Args:
    //   None.
    //
    // Returns:
    //   void.
    function rollLockedUpdate(): void {
      holdRollLockedCameraPose({
        camera,
        target: threeControls.target,
        rollLockAxis,
        heldEyeOffset,
      });
      ThreeTrackballControlsImpl.prototype.update.call(threeControls);
    }
    // target and camera.position are public, so a caller can move either between drags; holding the pose at the top of every update is what keeps that write off every frame the renderer draws.
    threeControls.update = rollLockedUpdate;
    return controls;
  }
  return controls;
}

// Holds the camera on the roll-locked pose its own framing implies: the eye banded off the lock axis, and the up vector the view direction from that eye and the lock axis determine. Every framing that enters the roll-locked controls comes through here - the one they are constructed on, the one a camera-sync peer writes through applyCameraState, and the one each drag step leaves behind - so the camera right axis is derived from a banded eye every time and never collapses onto the direction a view running parallel to the axis cannot name. An eye written exactly onto the target, or the target onto the eye, implies no framing at all, so there the lock holds the eye offset it last held around wherever the target now stands.
//
// Args:
//   camera: the perspective camera the controls drive; its position and up vector are rewritten in place, and it is left looking at the target.
//   target: the rotation target the eye offset is measured from, in the scene's own world frame.
//   rollLockAxis: the unit-length world-space axis camera roll is locked about.
//   heldEyeOffset: the eye offset these controls last held, as the camera position minus the rotation target in the scene's own world frame; read in place of an offset of zero length, and overwritten in place with the offset this call holds.
//
// Returns:
//   void.
function holdRollLockedCameraPose({
  camera,
  target,
  rollLockAxis,
  heldEyeOffset
}: {
  camera: THREE.PerspectiveCamera;
  target: THREE.Vector3;
  rollLockAxis: THREE.Vector3;
  heldEyeOffset: THREE.Vector3
}): void {
  let offset = camera.position.clone().sub(target);
  // A zero-length offset has no polar angle for the band to read, so it would pass through unbanded and cross into a zero right axis, a zero up vector, and a lookAt with no direction. The held offset keeps the view the lock last drew, and gives three's pan and zoom, which both scale by the eye distance, a distance to act on.
  if (offset.lengthSq() === 0) {
    offset = heldEyeOffset;
  }
  const bandedOffset = resolveRollLockBandedOffset({ offset, rollLockAxis });
  heldEyeOffset.copy(bandedOffset);
  const cameraRightAxis = new THREE.Vector3()
    .crossVectors(bandedOffset.clone().negate(), rollLockAxis)
    .normalize();
  camera.position.copy(target).add(bandedOffset);
  camera.up
    .crossVectors(cameraRightAxis, bandedOffset.clone().negate().normalize())
    .normalize();
  camera.lookAt(target);
  return;
}

// Bands an eye offset's polar angle off the lock axis into [ROLL_LOCKED_POLAR_ANGLE_EPSILON, pi - ROLL_LOCKED_POLAR_ANGLE_EPSILON], rebuilding it at the banded angle on its own meridian. Every camera right axis the roll-locked rotation derives comes from an offset this has already banded, so the collapse that derivation hits on an eye sitting exactly on the axis is unreachable rather than repaired afterwards.
//
// Args:
//   offset: the eye offset to band, as the camera position minus the rotation target in the scene's own world frame.
//   rollLockAxis: the unit-length world-space axis camera roll is locked about.
//
// Returns:
//   The offset at the same radius, banded off the lock axis; the offset itself when it already stands inside the band.
function resolveRollLockBandedOffset({
  offset,
  rollLockAxis
}: {
  offset: THREE.Vector3;
  rollLockAxis: THREE.Vector3
}): THREE.Vector3 {
  if (
    offset.angleTo(rollLockAxis) >= ROLL_LOCKED_POLAR_ANGLE_EPSILON &&
    offset.angleTo(rollLockAxis) <= Math.PI - ROLL_LOCKED_POLAR_ANGLE_EPSILON
  ) {
    return offset;
  }
  const meridian = resolveRollLockMeridian({ offset, rollLockAxis });
  const bandedPolarAngle = Math.min(
    Math.max(offset.angleTo(rollLockAxis), ROLL_LOCKED_POLAR_ANGLE_EPSILON),
    Math.PI - ROLL_LOCKED_POLAR_ANGLE_EPSILON,
  );
  const radius = offset.length();
  const bandedOffset = meridian
    .multiplyScalar(radius * Math.sin(bandedPolarAngle))
    .addScaledVector(rollLockAxis, radius * Math.cos(bandedPolarAngle));
  return bandedOffset;
}

// Resolves the meridian an eye offset stands on, as a unit vector perpendicular to the lock axis.
//
// Args:
//   offset: the eye offset whose meridian is read, as the camera position minus the rotation target in the scene's own world frame.
//   rollLockAxis: the unit-length world-space axis camera roll is locked about.
//
// Returns:
//   A fresh unit-length vector perpendicular to the lock axis.
function resolveRollLockMeridian({
  offset,
  rollLockAxis
}: {
  offset: THREE.Vector3;
  rollLockAxis: THREE.Vector3
}): THREE.Vector3 {
  const meridian = offset
    .clone()
    .addScaledVector(rollLockAxis, -offset.dot(rollLockAxis));
  if (meridian.lengthSq() > 0) {
    meridian.normalize();
    return meridian;
  }
  // An offset lying on the axis stands on every meridian at once, so it names none of its own and this is the one it is banded onto. Crossing the axis with the world basis vector it leans on least is what keeps this cross product itself clear of the collapse it stands in for.
  const axisMagnitudes = [
    Math.abs(rollLockAxis.x),
    Math.abs(rollLockAxis.y),
    Math.abs(rollLockAxis.z),
  ];
  const leastLeanedBasisVector = new THREE.Vector3().setComponent(
    axisMagnitudes.indexOf(Math.min(...axisMagnitudes)),
    1,
  );
  const fallbackMeridian = new THREE.Vector3()
    .crossVectors(rollLockAxis, leastLeanedBasisVector)
    .normalize();
  return fallbackMeridian;
}

// Validates the constructed controls satisfy every trackball contract by running the mouse-mapping, no-orbit, no-pose-clamp, and roll-lock assertions.
//
// Args:
//   controls: the constructed trackball controls.
//   camera: the perspective camera the controls drive.
//   renderer: the WebGL renderer whose canvas the controls listen on.
//   lockRoll: the world-space axis the controls were asked to lock roll about; null asserts the free trackball.
//
// Returns:
//   void.
function assertTrackballCameraControls({
  controls,
  camera,
  renderer,
  lockRoll
}: {
  controls: ThreeTrackballCameraControls;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  lockRoll: THREE.Vector3 | null
}): void {
  assertTrackballMouseMapping({ controls, renderer });
  assertNoOrbitCameraControls({ controls });
  assertNoCameraPoseClamps({ controls, lockRoll });
  assertRollLock({ controls, camera, lockRoll });
  return;
}

// Asserts the controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
//
// Args:
//   controls: the constructed trackball controls.
//   renderer: the WebGL renderer whose canvas the controls listen on.
//
// Returns:
//   void.
function assertTrackballMouseMapping({
  controls,
  renderer,
}: {
  controls: ThreeTrackballCameraControls;
  renderer: THREE.WebGLRenderer;
}): void {
  // Three's trackball fixes left-drag to rotation, right-drag to pan, and the wheel to zoom, so each mapping is live exactly when its disable flag is off; a roll-locked construction hands the left-drag to its own rotation instead, so the left-drag goes unmapped only when three's rotation is off and no roll-locked rotation replaced it.
  if ((controls.noRotate && controls.rollLockAxis === null) || controls.noPan || controls.noZoom) {
    throw new Error(
      `invalid trackball camera controls: noRotate=${controls.noRotate} noPan=${controls.noPan} noZoom=${controls.noZoom}`,
    );
  }
  // Context-menu suppression lives in a listener, so the only way to read it back is to put a cancelable contextmenu event through the canvas, whose dispatch returns false exactly when a listener prevented its default; every listener on it does nothing but preventDefault, so the probe leaves no state behind.
  if (renderer.domElement.dispatchEvent(new MouseEvent("contextmenu", { bubbles: false, cancelable: true }))) {
    throw new Error("context menu blocks trackball panning");
  }
  return;
}

// Asserts the controls do not use forbidden orbit-style target-locked camera semantics.
//
// Args:
//   controls: the constructed trackball controls.
//
// Returns:
//   void.
function assertNoOrbitCameraControls({
  controls,
}: {
  controls: ThreeTrackballCameraControls;
}): void {
  if (!(controls instanceof ThreeTrackballControlsImpl)) {
    throw new Error("orbit-style camera controls are forbidden");
  }
  return;
}

// Asserts the controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation beyond the polar band a roll lock costs.
//
// Args:
//   controls: the constructed trackball controls.
//   lockRoll: the world-space axis the controls were asked to lock roll about; null forbids every rotation and polar-angle restriction.
//
// Returns:
//   void.
function assertNoCameraPoseClamps({
  controls,
  lockRoll
}: {
  controls: ThreeTrackballCameraControls;
  lockRoll: THREE.Vector3 | null
}): void {
  // Three's trackball has no azimuth, target-lock, or translation clamp to read: its whole pose-restriction surface is the pan flag and the distance bounds, which stay at the unbounded defaults.
  if (
    controls.noPan ||
    controls.minDistance !== 0 ||
    controls.maxDistance !== Infinity
  ) {
    throw new Error(
      `restricted camera pose controls: noPan=${controls.noPan} minDistance=${controls.minDistance} maxDistance=${controls.maxDistance}`,
    );
  }
  // Three's own rotation carries the camera over a pole without stopping, so an unlocked path's polar angle is unrestricted exactly when that rotation is the one running and no roll-locked pitch clamp replaced it.
  if (lockRoll === null && (controls.noRotate || controls.rollLockPolarAngleEpsilon !== null)) {
    throw new Error(
      `restricted camera pose controls: noRotate=${controls.noRotate} rollLockPolarAngleEpsilon=${controls.rollLockPolarAngleEpsilon}`,
    );
  }
  // A supplied axis buys the roll lock at the polar extremes: the roll-locked pitch stops the camera at the pole rather than carrying the view through it, so the polar clamp is the lock's own price and not a restriction this assertion forbids.
  if (lockRoll !== null && controls.noRotate && controls.rollLockAxis === null) {
    throw new Error(
      "roll lock must cost only the roll axis and the polar extremes: three's rotation is off and no roll-locked rotation replaced it",
    );
  }
  return;
}

// Asserts roll is held about lockRoll when one is supplied and left free when none is, this module owning no axis of its own. A held roll is both halves of the invariant: the camera right axis perpendicular to the axis, and the camera up vector on the axis's own side rather than hanging the scene upside down.
//
// Args:
//   controls: the constructed trackball controls.
//   camera: the perspective camera the controls drive, whose up vector carries the side of the axis the scene hangs on.
//   lockRoll: the world-space axis the controls were asked to lock roll about; null asserts the camera right axis is constrained against no axis at all.
//
// Returns:
//   void.
function assertRollLock({
  controls,
  camera,
  lockRoll
}: {
  controls: ThreeTrackballCameraControls;
  camera: THREE.PerspectiveCamera;
  lockRoll: THREE.Vector3 | null
}): void {
  if (
    lockRoll !== null &&
    (controls.rollLockAxis === null || !controls.rollLockAxis.equals(lockRoll.clone().normalize()))
  ) {
    throw new Error(
      "roll-locked camera controls must keep the camera right axis perpendicular to the supplied axis",
    );
  }
  // Perpendicularity alone reads the same whichever way is up, so it passes a camera that pitched through the pole and hangs the scene inverted. The pitch clamp is what keeps the up vector on the axis's side, and the up vector the controls drive is what says it did.
  if (
    lockRoll !== null &&
    (controls.rollLockPolarAngleEpsilon === null || camera.up.dot(lockRoll.clone().normalize()) < 0)
  ) {
    throw new Error(
      `roll-locked camera controls must keep the camera up vector on the supplied axis's side: rollLockPolarAngleEpsilon=${controls.rollLockPolarAngleEpsilon} up . axis = ${camera.up.dot(lockRoll.clone().normalize())}`,
    );
  }
  if (
    lockRoll === null &&
    (controls.rollLockAxis !== null || controls.rollLockPolarAngleEpsilon !== null)
  ) {
    throw new Error(
      "free trackball camera controls must leave camera roll unconstrained",
    );
  }
  return;
}

function buildThreeTrackballCameraState({
  camera,
  controls,
}: {
  camera: THREE.PerspectiveCamera;
  controls: ThreeTrackballControlsImpl;
}): CameraState {
  const cameraState: CameraState = {
    intrinsics: {
      aspect: camera.aspect,
      far: camera.far,
      fov: camera.fov,
      near: camera.near,
      projection: "perspective-three",
    },
    extrinsics: {
      position: vectorToRecord(camera.position),
      quaternion: quaternionToRecord(camera.quaternion),
      target: vectorToRecord(controls.target),
      up: vectorToRecord(camera.up),
    },
    intr_convention: "three_trackball",
    extr_convention: "three_trackball",
    name: null,
    id: null,
  };
  return cameraState;
}

function applyThreeTrackballCameraState({
  camera,
  controls,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  controls: ThreeTrackballControlsImpl;
  cameraState: CameraState | null;
}): void {
  if (cameraState === null || cameraState.extr_convention !== "three_trackball") {
    return;
  }
  const position = cameraState.extrinsics.position;
  const quaternion = cameraState.extrinsics.quaternion;
  const target = cameraState.extrinsics.target;
  const up = cameraState.extrinsics.up;
  const aspect = cameraState.intrinsics.aspect;
  const far = cameraState.intrinsics.far;
  const fov = cameraState.intrinsics.fov;
  const near = cameraState.intrinsics.near;
  if (
    !isVectorRecord(position) ||
    !isQuaternionRecord(quaternion) ||
    !isVectorRecord(target) ||
    !isVectorRecord(up) ||
    typeof aspect !== "number" ||
    typeof far !== "number" ||
    typeof fov !== "number" ||
    typeof near !== "number"
  ) {
    return;
  }
  camera.position.set(position.x, position.y, position.z);
  camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
  camera.up.set(up.x, up.y, up.z);
  controls.target.set(target.x, target.y, target.z);
  camera.aspect = aspect;
  camera.far = far;
  camera.fov = fov;
  camera.near = near;
  camera.updateProjectionMatrix();
  controls.update();
}

function vectorToRecord(vector: THREE.Vector3): Record<string, number> {
  const vectorRecord = {
    x: vector.x,
    y: vector.y,
    z: vector.z,
  };
  return vectorRecord;
}

function quaternionToRecord(
  quaternion: THREE.Quaternion,
): Record<string, number> {
  const quaternionRecord = {
    x: quaternion.x,
    y: quaternion.y,
    z: quaternion.z,
    w: quaternion.w,
  };
  return quaternionRecord;
}

function isVectorRecord(value: unknown): value is {
  x: number;
  y: number;
  z: number;
} {
  const isVector = (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { x?: unknown }).x === "number" &&
    typeof (value as { y?: unknown }).y === "number" &&
    typeof (value as { z?: unknown }).z === "number"
  );
  return isVector;
}

function isQuaternionRecord(value: unknown): value is {
  x: number;
  y: number;
  z: number;
  w: number;
} {
  const isQuaternion = (
    isVectorRecord(value) &&
    typeof (value as { w?: unknown }).w === "number"
  );
  return isQuaternion;
}
