import type { ElementVNode, LeafVNode } from "web/reconcile/reconcile";
import type {
  SelectorResponse,
  SelectionNode,
} from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
import { completeRootLeafPath } from "data/viewer/utils/controls/selectors/ts/frontend/selection_path";

// Render one selector axis as a cascade of native <select> dropdowns: descend
// the response's imaginary root along the current path, one dropdown per level
// to a leaf; the app supplies only the option tree, the current path, and an
// onPathChange handler.
export function renderSelectorCascade({
  axisKey,
  response,
  path,
  onPathChange,
}: {
  axisKey: string;
  response: SelectorResponse;
  path: string[];
  onPathChange: (next: string[]) => void;
}): ElementVNode {
  const selectLeaves: LeafVNode[] = _renderSelectorLevel({
    node: response.root,
    level: 0,
    axisKey,
    path,
    onPathChange,
  });
  return {
    kind: "element",
    tag: "div",
    key: `${axisKey}-cascade`,
    props: {},
    children: selectLeaves,
  };
}

// Recursion helper: collect the <select> leaves from this level down; the base
// case (a node with no children) contributes none.
function _renderSelectorLevel({
  node,
  level,
  axisKey,
  path,
  onPathChange,
}: {
  node: SelectionNode;
  level: number;
  axisKey: string;
  path: string[];
  onPathChange: (next: string[]) => void;
}): LeafVNode[] {
  if (node.children.length === 0) {
    return [];
  }

  const selectedValue: string = path[level] ?? node.children[0].value;

  // The <select> change handler: report the completed root-leaf path to
  // onPathChange.
  const _onLevelChange = (event: Event): void => {
    const value: string = (event.target as HTMLSelectElement).value;
    onPathChange(completeRootLeafPath({ root: node, path, level, value }));
  };

  // The <select> is a reconciler leaf keyed by its option-set identity so a
  // coarser-level change re-mounts it with this parent's children.
  const selectLeaf: LeafVNode = {
    kind: "leaf",
    key: `${axisKey}-select-${level}-${path[level - 1] ?? "root"}`,
    props: { selectedValue },
    render: () => {
      const select: HTMLSelectElement = document.createElement("select");
      for (const child of node.children) {
        const option: HTMLOptionElement = document.createElement("option");
        option.value = child.value;
        option.textContent = child.label;
        select.appendChild(option);
      }
      select.value = selectedValue;
      select.addEventListener("change", _onLevelChange);
      return select;
    },
  };

  let selectedChild: SelectionNode = node.children[0];
  for (const child of node.children) {
    if (child.value === selectedValue) {
      selectedChild = child;
      break;
    }
  }
  const deeperLeaves: LeafVNode[] = _renderSelectorLevel({
    node: selectedChild,
    level: level + 1,
    axisKey,
    path,
    onPathChange,
  });
  return [selectLeaf, ...deeperLeaves];
}
