import * as THREE from "three";
import type { LeafVNode } from "web/reconcile/reconcile";
import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

// One spatial display response's part-B: build and return the THREE object the
// layered container adds to its shared scene.
export type SpatialLayerRenderer = ({
  displayResponse,
}: {
  displayResponse: DisplayResponse;
}) => THREE.Object3D;

// One raster display response's part-B: build and return the full-bleed node the
// layered container stacks.
export type RasterLayerRenderer = ({
  displayResponse,
}: {
  displayResponse: DisplayResponse;
}) => LeafVNode;

// display_kind -> spatial part-B; the module's single owner of the spatial
// registry, mutated only through the functions below.
const _spatialLayerRenderers = new Map<string, SpatialLayerRenderer>();

// display_kind -> raster part-B; the module's single owner of the raster
// registry, mutated only through the functions below.
const _rasterLayerRenderers = new Map<string, RasterLayerRenderer>();

// Register a spatial display kind's part-B so the layered container can build that
// kind's THREE object by display_kind lookup.
//
// Args:
//   displayKind: the spatial display_kind the part-B is registered under.
//   layerRenderer: the spatial part-B that builds a THREE object from a display
//     response of this kind.
//
// Returns:
//   void.
export function registerSpatialLayerRenderer({
  displayKind,
  layerRenderer,
}: {
  displayKind: string;
  layerRenderer: SpatialLayerRenderer;
}): void {
  _spatialLayerRenderers.set(displayKind, layerRenderer);
}

// Register a raster display kind's part-B so the layered container can build that
// kind's node by display_kind lookup.
//
// Args:
//   displayKind: the raster display_kind the part-B is registered under.
//   layerRenderer: the raster part-B that builds a node from a display response of
//     this kind.
//
// Returns:
//   void.
export function registerRasterLayerRenderer({
  displayKind,
  layerRenderer,
}: {
  displayKind: string;
  layerRenderer: RasterLayerRenderer;
}): void {
  _rasterLayerRenderers.set(displayKind, layerRenderer);
}

// Resolve the spatial part-B registered for a display kind, throwing when none is
// registered.
//
// Args:
//   displayKind: the spatial display_kind whose part-B is resolved.
//
// Returns:
//   The spatial part-B registered for displayKind.
export function getSpatialLayerRenderer({
  displayKind,
}: {
  displayKind: string;
}): SpatialLayerRenderer {
  const layerRenderer = _spatialLayerRenderers.get(displayKind);
  if (layerRenderer === undefined) {
    throw new Error(`no spatial layer renderer is registered for display kind: ${displayKind}`);
  }
  return layerRenderer;
}

// Resolve the raster part-B registered for a display kind, throwing when none is
// registered.
//
// Args:
//   displayKind: the raster display_kind whose part-B is resolved.
//
// Returns:
//   The raster part-B registered for displayKind.
export function getRasterLayerRenderer({
  displayKind,
}: {
  displayKind: string;
}): RasterLayerRenderer {
  const layerRenderer = _rasterLayerRenderers.get(displayKind);
  if (layerRenderer === undefined) {
    throw new Error(`no raster layer renderer is registered for display kind: ${displayKind}`);
  }
  return layerRenderer;
}
