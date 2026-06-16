import type { VNode } from "web/reconcile/reconcile";
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
}): VNode {
  return _renderSelectorLevel({
    node: response.root,
    level: 0,
    axisKey,
    path,
    onPathChange,
  });
}

// Recursion helper: render a <select> over this node's children, then recurse
// into the child the path selects, stopping at a leaf.
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
}): VNode {
  const children: VNode[] = [];
  if (node.children.length > 0) {
    const selectedValue: string = path[level] ?? node.children[0].value;

    // The <select> change handler: report the completed root-leaf path to
    // onPathChange.
    const _onLevelChange = (event: Event): void => {
      const value: string = (event.target as HTMLSelectElement).value;
      onPathChange(
        completeRootLeafPath({ root: node, path, level, value }),
      );
    };

    // The <select> is a reconciler leaf keyed by its option-set identity so a
    // coarser-level change re-mounts it with this parent's children.
    const selectLeaf: VNode = {
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
    children.push(selectLeaf);

    let selectedChild: SelectionNode = node.children[0];
    for (const child of node.children) {
      if (child.value === selectedValue) {
        selectedChild = child;
        break;
      }
    }
    children.push(
      _renderSelectorLevel({
        node: selectedChild,
        level: level + 1,
        axisKey,
        path,
        onPathChange,
      }),
    );
  }
  return {
    kind: "element",
    tag: "div",
    key: `${axisKey}-level-${level}`,
    props: {},
    children,
  };
}
