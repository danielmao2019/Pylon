export interface DisplayResponse {
  slot_id: string;
  title: string;
  display_kind: string;
  url: string | null;
  original_overlay_url?: string | null;
  meta_info: Record<string, unknown>;
}
