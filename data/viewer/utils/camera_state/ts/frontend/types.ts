export interface CameraState {
  intrinsics: Record<string, unknown>;
  extrinsics: Record<string, unknown>;
  convention: string;
  name: string | null;
  id: string | null;
}
