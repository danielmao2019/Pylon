export interface CameraState {
  intrinsics: Record<string, unknown>;
  extrinsics: Record<string, unknown>;
  intr_convention: string;
  extr_convention: string;
  name: string | null;
  id: string | null;
}
