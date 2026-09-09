// Node harness driving the three.js trackball camera controls through a scripted drag.
//
// The harness stands in for the browser: it builds a jsdom document, constructs the
// shipped controls over a real perspective camera, and puts pointer events through the
// canvas the way a left-drag does. Nothing about the roll lock itself is modelled here;
// the pitch, the yaw, and the clamp all run inside the module under test.
//
// Usage: tsx roll_lock_harness.ts '<spec-json>', where the spec carries `lock_roll`,
// `position`, `target`, `up`, and `drags`. One JSON record for the constructed camera
// plus one per drag is written to stdout.

import { JSDOM } from "jsdom";
import * as THREE from "three";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  pretendToBeVisual: true,
});
const domWindow = dom.window as unknown as Window & typeof globalThis;
const globalScope = globalThis as unknown as Record<string, unknown>;
// The module reaches for the browser globals through the bare names, so the jsdom window
// is installed under those names before the module is loaded.
for (const globalName of [
  "document",
  "navigator",
  "Element",
  "HTMLElement",
  "HTMLCanvasElement",
  "Event",
  "MouseEvent",
  "PointerEvent",
  "WheelEvent",
  "MutationObserver",
  "requestAnimationFrame",
  "cancelAnimationFrame",
  "getComputedStyle",
]) {
  globalScope[globalName] = (domWindow as unknown as Record<string, unknown>)[globalName];
}
globalScope["window"] = domWindow;

import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";

interface DragSpec {
  dx: number;
  dy: number;
}

interface HarnessSpec {
  lock_roll: [number, number, number] | null;
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  drags: DragSpec[];
}

const spec = JSON.parse(process.argv[2]) as HarnessSpec;

const container = domWindow.document.createElement("div");
domWindow.document.body.appendChild(container);
const canvas = domWindow.document.createElement("canvas");
container.appendChild(canvas);
// The controls read nothing off the renderer but the canvas they listen on, and a
// WebGL context cannot exist outside a browser, so the canvas is handed over on its own.
const renderer = { domElement: canvas } as unknown as THREE.WebGLRenderer;

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.position.set(spec.position[0], spec.position[1], spec.position[2]);
camera.up.set(spec.up[0], spec.up[1], spec.up[2]);
const controls = createTrackballCameraControls({
  container,
  camera,
  renderer,
  initialCameraState: null,
  lockRoll:
    spec.lock_roll === null
      ? null
      : new THREE.Vector3(spec.lock_roll[0], spec.lock_roll[1], spec.lock_roll[2]),
});
controls.target.set(spec.target[0], spec.target[1], spec.target[2]);

const axis =
  spec.lock_roll === null
    ? new THREE.Vector3(0, 0, 1)
    : new THREE.Vector3(spec.lock_roll[0], spec.lock_roll[1], spec.lock_roll[2]).normalize();

// Reads the roll-lock invariants off the camera the drag left behind: the camera right
// axis's component along the lock axis, the up vector's side of it, and the polar angle
// that says how close to the pole the camera stands.
function measureCamera(): Record<string, unknown> {
  const offset = camera.position.clone().sub(controls.target);
  const forward = offset.clone().negate().normalize();
  const cameraRightAxis = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const components = camera.position
    .toArray()
    .concat(camera.up.toArray())
    .concat(cameraRightAxis.toArray());
  return {
    right_along_axis: cameraRightAxis.dot(axis),
    up_along_axis: camera.up.clone().normalize().dot(axis),
    up_length: camera.up.length(),
    camera_right_axis_length: cameraRightAxis.length(),
    polar: offset.angleTo(axis),
    position: camera.position.toArray(),
    up: camera.up.toArray(),
    camera_right_axis: cameraRightAxis.toArray(),
    finite: components.every((component) => Number.isFinite(component)),
  };
}

let clientX = 400;
let clientY = 300;

// Puts one left-drag step through the canvas: the press opens the drag on the canvas the
// controls listen on, and each move is delivered on the window, where the module tracks
// the pointer once the drag is open.
function dispatchPointer(target: EventTarget, type: string): void {
  target.dispatchEvent(
    new domWindow.MouseEvent(type, {
      button: 0,
      buttons: 1,
      clientX,
      clientY,
      bubbles: true,
      cancelable: true,
    }),
  );
}

const records: Array<Record<string, unknown>> = [measureCamera()];
dispatchPointer(canvas, "pointerdown");
for (const drag of spec.drags) {
  clientX += drag.dx;
  clientY += drag.dy;
  dispatchPointer(domWindow, "pointermove");
  records.push(measureCamera());
}
dispatchPointer(domWindow, "pointerup");

process.stdout.write(
  JSON.stringify({
    roll_lock_axis: controls.rollLockAxis === null ? null : controls.rollLockAxis.toArray(),
    roll_lock_polar_angle_epsilon: controls.rollLockPolarAngleEpsilon,
    records,
  }),
);
