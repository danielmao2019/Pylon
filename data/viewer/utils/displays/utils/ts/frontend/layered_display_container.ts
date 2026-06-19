import * as THREE from "three";
import type { ElementVNode, LeafVNode, VNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { LayeredDisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/layered_display_response";
import {
  getSpatialLayerRenderer,
  getRasterLayerRenderer,
} from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
// Side-effect: eager-glob-loads every modality so its self-registration populates
// the registry before any render.
import "data/viewer/utils/displays/utils/ts/frontend/register_layer_renderers";
import {
  createSpatialDisplayScene,
  startThreeSceneRenderLoop,
  attachThreeScenePickSeam,
} from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";

// Composes one layered display response into a shared spatial WebGL scene or a
// stacked raster DOM container per cell, routing on the backend-stamped
// layer_class.
//
// Args:
//   layeredDisplayResponse: the layered response carrying its base + aux layers
//     and the backend-stamped layer_class that selects the spatial/raster branch.
//   initialCameraState: initial framing for the spatial branch's one shared camera
//     (camera-to-world extrinsics + intrinsics); null uses the camera's default
//     framing.
//
// Returns:
//   The container VNode for the routed layer class.
export function renderLayeredDisplay({
  layeredDisplayResponse,
  initialCameraState,
}: {
  layeredDisplayResponse: LayeredDisplayResponse;
  initialCameraState: CameraState | null;
}): VNode {
  if (layeredDisplayResponse.layer_class === "spatial") {
    return renderLayeredSpatialDisplay({ layeredDisplayResponse, initialCameraState });
  }
  if (layeredDisplayResponse.layer_class === "raster") {
    return renderLayeredRasterDisplay({ layeredDisplayResponse });
  }
  throw new Error(
    `layered display response has an unknown layer class: ${JSON.stringify(layeredDisplayResponse.layer_class)}`,
  );
}

// Renders the base + aux spatial layers into one shared scene/camera as a
// slot_id-keyed LeafVNode, the shared camera owning the framing and the additive
// pick seam.
//
// Args:
//   layeredDisplayResponse: the layered response whose base + aux layers are built
//     into the one shared scene.
//   initialCameraState: initial framing for the one shared camera (camera-to-world
//     extrinsics + intrinsics); null uses the camera's default framing.
//
// Returns:
//   A LeafVNode keyed by layeredDisplayResponse.slot_id whose render() mounts the
//   shared spatial context.
function renderLayeredSpatialDisplay({
  layeredDisplayResponse,
  initialCameraState,
}: {
  layeredDisplayResponse: LayeredDisplayResponse;
  initialCameraState: CameraState | null;
}): VNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: layeredDisplayResponse.slot_id,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createSpatialDisplayScene({
        initialCameraState,
      });
      const layerObjects = createLayerObjects({ layeredDisplayResponse });
      layerObjects.forEach((object) => scene.add(object));
      const controls = createTrackballCameraControls({
        container,
        camera,
        renderer,
        initialCameraState,
      });
      _registerSceneResize({ container, camera, renderer, controls });
      _syncCameraState({ container, controls });
      attachThreeScenePickSeam({ container, camera, scenes: [scene] });
      renderLayeredSpatialScene({ scene, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

// Builds the THREE object for every layer by dispatching each layer's display
// response to its registry-resolved spatial renderer.
//
// Args:
//   layeredDisplayResponse: the layered response whose base + aux layers are each
//     resolved to a spatial part-B by display_kind and built into a THREE object.
//
// Returns:
//   The THREE objects for the base + aux layers, in layer order.
function createLayerObjects({
  layeredDisplayResponse,
}: {
  layeredDisplayResponse: LayeredDisplayResponse;
}): THREE.Object3D[] {
  const layerObjects: THREE.Object3D[] = [];
  for (const layer of [
    layeredDisplayResponse.base_display_response,
    ...layeredDisplayResponse.aux_display_responses,
  ]) {
    const layerRenderer = getSpatialLayerRenderer({ displayKind: layer.display_kind });
    layerObjects.push(layerRenderer({ displayResponse: layer }));
  }
  return layerObjects;
}

// Drives the shared layered-scene render loop with the base-camera trackball
// controls.
//
// Args:
//   scene: the one shared scene every layer's object was added into.
//   camera: the one shared camera that owns the framing.
//   renderer: the one shared renderer.
//   controls: the one shared camera's trackball controls, updated each frame.
//
// Returns:
//   void.
function renderLayeredSpatialScene({
  scene,
  camera,
  renderer,
  controls,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  startThreeSceneRenderLoop({ scene, camera, renderer, controls });
}

// Stacks the base + aux raster layer nodes as absolutely-positioned full-bleed
// elements keyed by layer slot_id, each produced by its registry-resolved raster
// part-B.
//
// Args:
//   layeredDisplayResponse: the layered response whose base + aux layers are each
//     resolved to a raster part-B by display_kind and stacked full-bleed.
//
// Returns:
//   The relative container ElementVNode whose children are the stacked layers.
function renderLayeredRasterDisplay({
  layeredDisplayResponse,
}: {
  layeredDisplayResponse: LayeredDisplayResponse;
}): VNode {
  const children: VNode[] = [];
  for (const layer of [
    layeredDisplayResponse.base_display_response,
    ...layeredDisplayResponse.aux_display_responses,
  ]) {
    const layerRenderer = getRasterLayerRenderer({ displayKind: layer.display_kind });
    const cell: ElementVNode = {
      kind: "element",
      tag: "div",
      key: `${layeredDisplayResponse.slot_id}/layer/${layer.slot_id}`,
      props: {
        style: { position: "absolute", inset: "0", width: "100%", height: "100%" },
      },
      children: [layerRenderer({ displayResponse: layer })],
    };
    children.push(cell);
  }
  const container: ElementVNode = {
    kind: "element",
    tag: "div",
    key: layeredDisplayResponse.slot_id,
    props: {
      className: "layered-display-container",
      style: { position: "relative", width: "100%", height: "100%" },
    },
    children,
  };
  return container;
}

// Keeps the renderer size + camera aspect synced to the one shared container via a
// ResizeObserver — the layered container's own copy of the per-display resize
// helper.
//
// Args:
//   container: the one shared display container the renderer is sized against.
//   camera: the one shared camera whose aspect tracks the container.
//   renderer: the one shared renderer resized to the container.
//   controls: the shared camera's trackball controls, notified of resizes.
//
// Returns:
//   void.
function _registerSceneResize({
  container,
  camera,
  renderer,
  controls,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  const resize = (): void => {
    const width = Math.max(1, container.clientWidth || 1);
    const height = Math.max(1, container.clientHeight || 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    controls.handleResize();
  };
  resize();
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(resize).observe(container);
  }
  window.addEventListener("resize", resize);
}

// Publishes this cell's shared-camera pose now and re-publishes on every controls
// change, so other cells can observe and sync to it.
//
// Args:
//   container: the one shared display container the camera state is published onto.
//   controls: the shared camera's trackball controls supplying the camera state.
//
// Returns:
//   void.
function _syncCameraState({
  container,
  controls,
}: {
  container: HTMLDivElement;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  _publishCameraState({ container, controls });
  controls.addEventListener("change", () => {
    _publishCameraState({ container, controls });
  });
}

// Publishes the controls' shared-camera state onto the container (dataset.cameraState
// plus a bubbling camera-pose-change event) so the consumer can persist this cell's
// camera pose — the layered container's copy of the per-display publish helper.
//
// Args:
//   container: the one shared display container the state is published onto.
//   controls: the shared camera's trackball controls supplying the camera state.
//
// Returns:
//   void.
function _publishCameraState({
  container,
  controls,
}: {
  container: HTMLDivElement;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  const cameraState = controls.getCameraState();
  if (cameraState === null) {
    return;
  }
  container.dataset.cameraState = JSON.stringify(cameraState);
  container.dispatchEvent(
    new CustomEvent<CameraState>("camera-pose-change", {
      bubbles: true,
      detail: cameraState,
    }),
  );
}
