import type { SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";

// Complete a selector level change into a full root-leaf path: the prefix up to
// the chosen level, the chosen value, then each deeper level's first child
// descended to a leaf — so a non-leaf choice resets every finer level to its
// first option.
export function completeRootLeafPath({
  root,
  path,
  level,
  value,
}: {
  root: SelectionNode;
  path: string[];
  level: number;
  value: string;
}): string[] {
  const completedPath: string[] = path.slice(0, level).concat([value]);
  let node: SelectionNode =
    root.children.find((child) => child.value === value) ?? root.children[0];
  while (node.children.length > 0) {
    const firstChild: SelectionNode = node.children[0];
    completedPath.push(firstChild.value);
    node = firstChild;
  }
  return completedPath;
}
