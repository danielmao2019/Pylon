import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";

export interface VideoDisplayResponse extends DisplayResponse {
  display_kind: "video";
}
