import * as THREE from "three";
import type { ElementVNode, LeafVNode, VNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import {
  createThreeDisplayContainer,
  createThreePerspectiveCamera,
  createThreeScene,
  createThreeWebGLRenderer,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";

// A spatial layer builds its THREE object(s) into its OWN scene; one scene per
// layer lets the render loop composite layers without Z-fighting coincident
// geometry.
export type SpatialLayerContributor = (scene: THREE.Scene) => void;

// Every layer (base + ALL aux) is passed regardless of visibility — `id`
// identifies it, `visible` is its current toggle state; the consumer rebuilds
// this list each render with stable `id`s and updated `visible` flags. Each
// layer carries its class-appropriate render seam — a spatial scene-contributor
// or a raster 2D node.
export type LayerSpec =
  | { id: string; visible: boolean; displayClass: "spatial"; contributeToScene: SpatialLayerContributor }
  | { id: string; visible: boolean; displayClass: "raster"; node: VNode };

// Composes a layered display into ONE WebGL context per cell; routes on the
// backend-stamped class (layers[0].displayClass, homogeneous per
// layered_display_response.model_post_init), so the base layer's displayClass is
// representative.
//
// Args:
//   layers: the layer specs, homogeneous in displayClass by backend guarantee;
//     the base layer is index 0, aux layers follow in array order.
//   slotId: stable key for the container VNode.
//   initialCameraState: initial framing for the spatial branch's one base
//     camera (camera-to-world extrinsics + intrinsics); null lets the camera use
//     its default framing.
//
// Returns:
//   The container VNode for the routed layer class.
export function renderLayeredDisplayContainer({
  layers,
  slotId,
  initialCameraState,
}: {
  layers: readonly LayerSpec[];
  slotId: string;
  initialCameraState: CameraState | null;
}): VNode {
  if (layers[0].displayClass === "spatial") {
    return renderSpatialLayeredContainer({ layers, slotId, initialCameraState });
  }
  if (layers[0].displayClass === "raster") {
    return renderRasterLayeredContainer({ layers, slotId });
  }
  throw new Error(
    `layered display container received an unknown layer class: ${JSON.stringify(layers[0])}`,
  );
}

// Renders the spatial layers into ONE shared context as a LeafVNode keyed by the
// STABLE slotId (never re-mounts on toggle) — the base layer owns the
// camera/controls and aux follow it; the visible set rides as a
// data-visible-layers prop the render loop reads each frame so only the base-camera
// pose is published for the consumer to observe (e.g. persist it across the
// column's mode cells).
//
// Args:
//   layers: the layer specs; the first is the base layer, the rest are aux
//     layers (the base owns the camera/controls).
//   slotId: stable key for the container LeafVNode.
//   initialCameraState: initial framing for the one base camera (camera-to-world
//     extrinsics + intrinsics); null uses the camera's default framing.
//
// Returns:
//   A LeafVNode keyed by slotId whose render() mounts the shared spatial context.
function renderSpatialLayeredContainer({
  layers,
  slotId,
  initialCameraState,
}: {
  layers: readonly LayerSpec[];
  slotId: string;
  initialCameraState: CameraState | null;
}): VNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: slotId,
    props: {
      "data-visible-layers": layers
        .filter((layer) => layer.visible)
        .map((layer) => layer.id)
        .join(","),
    },
    render: () => {
      const { container, baseScene, auxScenes, camera, renderer } = createLayeredSpatialScene({
        layers,
        initialCameraState,
      });
      const controls = createTrackballCameraControls({ container, camera, renderer, initialCameraState });
      _registerSceneResize({ container, camera, renderer, controls });
      _publishCameraState({ container, controls });
      controls.addEventListener("change", () => {
        _publishCameraState({ container, controls });
      });
      renderLayeredSpatialScene({ container, baseScene, auxScenes, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

// Composes the one shared container/camera/renderer, then one THREE.Scene per
// layer (layers[0] = base, the rest = aux), each populated by its contributor.
//
// Args:
//   layers: the spatial layer specs; the first is the base layer, the rest are
//     aux layers, each carrying a scene-contributor and stable id.
//   initialCameraState: initial framing for the one base camera (camera-to-world
//     extrinsics + intrinsics); null uses the camera's default framing.
//
// Returns:
//   The shared container, the base scene, the id-tagged aux scenes, the one base
//   camera, and the one renderer.
function createLayeredSpatialScene({
  layers,
  initialCameraState,
}: {
  layers: readonly LayerSpec[];
  initialCameraState: CameraState | null;
}): {
  container: HTMLDivElement;
  baseScene: THREE.Scene;
  auxScenes: { id: string; scene: THREE.Scene }[];
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
} {
  const container = createThreeDisplayContainer({ pointerEventsSuppressed: false });
  const camera = createThreePerspectiveCamera({ initialCameraState });
  const renderer = createThreeWebGLRenderer({ container });
  const baseScene = createThreeScene();
  (layers[0] as { contributeToScene: SpatialLayerContributor }).contributeToScene(baseScene);
  const auxScenes = layers.slice(1).map((layer) => {
    const scene = createThreeScene();
    (layer as { contributeToScene: SpatialLayerContributor }).contributeToScene(scene);
    return { id: layer.id, scene };
  });
  return { container, baseScene, auxScenes, camera, renderer };
}

// Reuses startThreeSceneRenderLoop for the base pass; in onAfterRender re-reads
// container.dataset.visibleLayers each frame and draws only the listed aux
// scenes over the base (clearDepth between), so toggling the attribute toggles a
// layer with no observer.
//
// Args:
//   container: the one shared display container whose data-visible-layers
//     attribute is re-read each frame to select which aux scenes draw.
//   baseScene: the base layer's scene, drawn by the loop's auto-clearing pass.
//   auxScenes: the id-tagged aux layers' scenes, drawn on top per frame after the
//     base when their id is in the visible set.
//   camera: the one base camera shared by every pass.
//   renderer: the one renderer shared by every pass.
//   controls: the base layer's trackball controls, updated each frame by the loop.
//
// Returns:
//   void.
function renderLayeredSpatialScene({
  container,
  baseScene,
  auxScenes,
  camera,
  renderer,
  controls,
}: {
  container: HTMLElement;
  baseScene: THREE.Scene;
  auxScenes: readonly { id: string; scene: THREE.Scene }[];
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  const onAfterRender = (): void => {
    const ids = new Set(
      (container.dataset.visibleLayers ?? "").split(",").filter((s) => s.length > 0),
    );
    renderer.autoClear = false;
    for (const { id, scene } of auxScenes) {
      if (ids.has(id)) {
        renderer.clearDepth();
        renderer.render(scene, camera);
      }
    }
    renderer.autoClear = true;
  };
  startThreeSceneRenderLoop({ scene: baseScene, camera, renderer, controls, onAfterRender });
}

// Stacks the VISIBLE raster (2D image/video) layer nodes as absolutely-positioned
// full-bleed elements; each node is keyed by its layer id so the reconciler
// adds/removes only the toggled raster layers — DOM stacking has no
// shared-context cost, so raster needs no one-context observer.
//
// Args:
//   layers: the raster layer specs; only the visible ones are stacked, base
//     first, aux above in array order.
//   slotId: stable key for the container ElementVNode and its per-layer cells.
//
// Returns:
//   The relative container ElementVNode whose children are the stacked visible
//   layers.
function renderRasterLayeredContainer({
  layers,
  slotId,
}: {
  layers: readonly LayerSpec[];
  slotId: string;
}): VNode {
  const children: VNode[] = layers
    .filter((layer) => layer.visible)
    .map((layer) => {
      const cell: ElementVNode = {
        kind: "element",
        tag: "div",
        key: `${slotId}/layer/${layer.id}`,
        props: {
          className: "layered-display-container__layer",
          style: { position: "absolute", inset: "0", width: "100%", height: "100%" },
        },
        children: [(layer as { node: VNode }).node],
      };
      return cell;
    });
  const container: ElementVNode = {
    kind: "element",
    tag: "div",
    key: slotId,
    props: {
      className: "layered-display-container",
      style: { position: "relative", width: "100%", height: "100%" },
    },
    children,
  };
  return container;
}

// Keeps the renderer size + camera aspect synced to the one shared container via
// a ResizeObserver — the layered container's own copy of the per-display resize
// helper.
//
// Args:
//   container: the one shared display container the renderer is sized against.
//   camera: the one base camera whose aspect tracks the container.
//   renderer: the one renderer resized to the container.
//   controls: the base layer's trackball controls, notified of resizes.
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

// Publishes the controls' base-camera state onto the container — dataset.cameraState
// plus a bubbling camera-pose-change event — so the consumer can observe this
// cell's base-camera pose (e.g. persist it across mode cells); the layered
// container's own copy of the per-display publish helper.
//
// Args:
//   container: the one shared display container the state is published onto.
//   controls: the base layer's trackball controls supplying the camera state.
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
