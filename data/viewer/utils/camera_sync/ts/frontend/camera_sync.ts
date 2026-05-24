import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { CameraSyncState } from "./types";

type CameraSyncListener = (cameraSyncState: CameraSyncState) => void;

let cameraSyncState: CameraSyncState = {
  source_id: null,
  target_ids: [],
  camera_state: null,
};
const targetElements = new Map<string, HTMLElement>();
const listeners: CameraSyncListener[] = [];

export function loadCameraSyncState(cameraState: CameraState | null): void {
  cameraSyncState = {
    source_id: null,
    target_ids: readTargetIds(),
    camera_state: cameraState,
  };
  applyCameraStateToRegisteredTargets(cameraSyncState.camera_state, null);
  emitCameraSyncState();
}

export function getCameraSyncState(): CameraSyncState {
  return cameraSyncState;
}

export function subscribeCameraSyncState(
  listener: CameraSyncListener,
): () => void {
  listeners.push(listener);
  return () => {
    const index = listeners.indexOf(listener);
    if (index >= 0) {
      listeners.splice(index, 1);
    }
  };
}

export function registerCameraSyncTarget(
  targetId: string,
  targetElement: HTMLElement,
): void {
  if (targetId.length === 0) {
    throw new Error("camera sync target id is empty");
  }
  targetElements.set(targetId, targetElement);
  cameraSyncState = {
    ...cameraSyncState,
    target_ids: readTargetIds(),
  };
  applyCameraStateToElement(targetElement, cameraSyncState.camera_state);
}

export function unregisterCameraSyncTarget(targetId: string): void {
  targetElements.delete(targetId);
  cameraSyncState = {
    ...cameraSyncState,
    target_ids: readTargetIds(),
  };
}

export function applyCameraSyncStateToTargets(cameraState: CameraState): void {
  cameraSyncState = {
    source_id: null,
    target_ids: readTargetIds(),
    camera_state: cameraState,
  };
  applyCameraStateToRegisteredTargets(cameraState, null);
  emitCameraSyncState();
}

export function applySourceCameraStateToTargets(
  sourceId: string,
  cameraState: CameraState,
): void {
  if (!targetElements.has(sourceId)) {
    throw new Error(`source camera sync target is not registered: ${sourceId}`);
  }
  cameraSyncState = {
    source_id: sourceId,
    target_ids: readTargetIds(),
    camera_state: cameraState,
  };
  applyCameraStateToRegisteredTargets(cameraState, sourceId);
  emitCameraSyncState();
}

function readTargetIds(): string[] {
  return Array.from(targetElements.keys());
}

function applyCameraStateToRegisteredTargets(
  cameraState: CameraState | null,
  sourceId: string | null,
): void {
  for (const [targetId, targetElement] of targetElements) {
    if (targetId === sourceId) {
      continue;
    }
    applyCameraStateToElement(targetElement, cameraState);
  }
}

function applyCameraStateToElement(
  targetElement: HTMLElement,
  cameraState: CameraState | null,
): void {
  if (cameraState === null) {
    delete targetElement.dataset.cameraState;
    return;
  }
  targetElement.dataset.cameraState = JSON.stringify(cameraState);
}

function emitCameraSyncState(): void {
  for (const listener of listeners) {
    listener(cameraSyncState);
  }
}
