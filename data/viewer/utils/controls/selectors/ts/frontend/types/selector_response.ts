// One selector axis: the imaginary root of its option tree — mirrors the
// backend SelectorResponse schema.
export interface SelectorResponse {
  root: SelectionNode;
}

// One option node of a selector axis: value, label, and child nodes (empty at a
// leaf) — mirrors the backend SelectionNode schema.
export interface SelectionNode {
  value: string;
  label: string;
  children: SelectionNode[];
}
