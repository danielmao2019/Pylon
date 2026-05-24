import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";

export interface PlaceholderDisplayResponse extends DisplayResponse {
  display_kind: "placeholder";
  message: string;
}
