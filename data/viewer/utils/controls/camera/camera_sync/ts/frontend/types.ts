import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";

export interface CameraSyncState {
  source_id: string | null;
  target_ids: readonly string[];
  camera_state: CameraState | null;
}
