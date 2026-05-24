import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { CameraSyncState } from "./types";

type CameraSyncListener = (cameraSyncState: CameraSyncState) => void;

export class CameraSyncRegistry {
  // Per-source camera-sync registry: each source_id owns an independent
  // CameraSyncState and target element pool, so apply operations stay
  // confined to their source's own pool.
  private _state_by_source_id: Record<string, CameraSyncState> = {};
  private _targets_by_source_id: Record<string, Map<string, HTMLElement>> = {};
  private _listeners: CameraSyncListener[] = [];

  loadCameraSyncState(sourceId: string, cameraState: CameraState | null): void {
    this._state_by_source_id[sourceId] = {
      source_id: sourceId,
      target_ids: [],
      camera_state: cameraState,
    };
    this._targets_by_source_id[sourceId] = new Map();
  }

  getCameraSyncState(sourceId: string): CameraSyncState | undefined {
    return this._state_by_source_id[sourceId];
  }

  subscribeCameraSyncState(listener: CameraSyncListener): () => void {
    this._listeners.push(listener);
    return () => {
      const index = this._listeners.indexOf(listener);
      if (index >= 0) {
        this._listeners.splice(index, 1);
      }
    };
  }

  registerCameraSyncTarget(
    sourceId: string,
    targetId: string,
    targetElement: HTMLElement,
  ): void {
    if (this._targets_by_source_id[sourceId] === undefined) {
      this._targets_by_source_id[sourceId] = new Map();
    }
    if (this._state_by_source_id[sourceId] === undefined) {
      this._state_by_source_id[sourceId] = {
        source_id: sourceId,
        target_ids: [],
        camera_state: null,
      };
    }
    this._targets_by_source_id[sourceId].set(targetId, targetElement);
    this._state_by_source_id[sourceId] = {
      ...this._state_by_source_id[sourceId],
      target_ids: Array.from(this._targets_by_source_id[sourceId].keys()),
    };
    this._apply_camera_state_to_element(
      targetElement,
      this._state_by_source_id[sourceId].camera_state,
    );
  }

  unregisterCameraSyncTarget(sourceId: string, targetId: string): void {
    const targets = this._targets_by_source_id[sourceId];
    if (targets === undefined) {
      return;
    }
    targets.delete(targetId);
    this._state_by_source_id[sourceId] = {
      ...this._state_by_source_id[sourceId],
      target_ids: Array.from(targets.keys()),
    };
  }

  applyCameraSyncStateToTargets(
    sourceId: string,
    cameraState: CameraState,
  ): void {
    if (this._targets_by_source_id[sourceId] === undefined) {
      this._targets_by_source_id[sourceId] = new Map();
    }
    const targets = this._targets_by_source_id[sourceId];
    const targetIds = Array.from(targets.keys());
    this._state_by_source_id[sourceId] = {
      source_id: sourceId,
      target_ids: targetIds,
      camera_state: cameraState,
    };
    for (const [, targetElement] of targets) {
      this._apply_camera_state_to_element(targetElement, cameraState);
    }
    this._emit_camera_sync_state(this._state_by_source_id[sourceId]);
  }

  applySourceCameraStateToTargets(
    sourceId: string,
    cameraState: CameraState,
  ): void {
    const targets = this._targets_by_source_id[sourceId];
    if (targets === undefined) {
      throw new Error(`camera sync source is not registered: ${sourceId}`);
    }
    const targetIds = Array.from(targets.keys());
    this._state_by_source_id[sourceId] = {
      source_id: sourceId,
      target_ids: targetIds,
      camera_state: cameraState,
    };
    for (const [targetId, targetElement] of targets) {
      if (targetId === sourceId) {
        continue;
      }
      this._apply_camera_state_to_element(targetElement, cameraState);
    }
    this._emit_camera_sync_state(this._state_by_source_id[sourceId]);
  }

  private _apply_camera_state_to_element(
    targetElement: HTMLElement,
    cameraState: CameraState | null,
  ): void {
    if (cameraState === null) {
      delete targetElement.dataset.cameraState;
      return;
    }
    targetElement.dataset.cameraState = JSON.stringify(cameraState);
  }

  private _emit_camera_sync_state(cameraSyncState: CameraSyncState): void {
    for (const listener of this._listeners) {
      listener(cameraSyncState);
    }
  }
}

export const cameraSyncRegistry = new CameraSyncRegistry();
