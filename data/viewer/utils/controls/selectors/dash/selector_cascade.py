"""Dash cascade selector rendered from an app's option tree."""

from typing import List, Tuple

from dash import dcc, html
from dash.development.base_component import Component

# One option subtree (value, label, children) — the app's native option
# representation, walked directly server-side; Dash has no request/response
# boundary, so there is no response envelope.
SelectionTree = Tuple[str, str, List["SelectionTree"]]


def render_selector_cascade(option_tree: SelectionTree, path: List[str]) -> Component:
    """Render one selector axis as a Dash cascade of dropdowns.

    Renders from the app's option tree and the current path: one dropdown per
    level, descending the imaginary-root option subtree along the path to a
    leaf, re-rendered per parent change.

    Args:
        option_tree: The imaginary-root ``(value, label, children)`` subtree
            where ``value`` and ``label`` are strings and ``children`` is a list
            of nested ``(value, label, children)`` subtrees.
        path: The current root-leaf selection path as a list of chosen values,
            one per descended level below the imaginary root.

    Returns:
        The dropdown-stack Dash component.
    """
    assert isinstance(
        option_tree, tuple
    ), "Option tree must be a tuple. option_tree=%r" % (option_tree,)
    assert isinstance(path, list), "Path must be a list. path=%r" % (path,)
    return _render_selector_level(node=option_tree, level=0, path=path)


def _render_selector_level(
    node: SelectionTree, level: int, path: List[str]
) -> Component:
    """Render this level's dropdown then recurse into the selected child.

    Recursion helper: a Dash dropdown over this node's child subtrees, then
    recurse into the child the path selects, stopping at a leaf.

    Args:
        node: The ``(value, label, children)`` subtree rooted at this level.
        level: Depth of this node below the imaginary root; ``0`` is the
            imaginary root, whose children populate the first dropdown.
        path: The current root-leaf selection path as a list of chosen values,
            one per descended level below the imaginary root.

    Returns:
        This level's dropdown plus the deeper stack, or an empty container at a
        leaf.
    """
    assert isinstance(node, tuple), "Node must be a tuple. node=%r" % (node,)
    assert isinstance(level, int), "Level must be an int. level=%r" % (level,)
    assert isinstance(path, list), "Path must be a list. path=%r" % (path,)
    _value, _label, children = node
    if children:
        selected_value = path[level] if level < len(path) else children[0][0]
        selected_child = children[0]
        for child in children:
            child_value, _child_label, _child_children = child
            if child_value == selected_value:
                selected_child = child
                break
        dropdown = dcc.Dropdown(
            options=[
                {"value": child_value, "label": child_label}
                for (child_value, child_label, _child_children) in children
            ],
            value=selected_value,
            clearable=False,
        )
        return html.Div(
            [
                dropdown,
                _render_selector_level(node=selected_child, level=level + 1, path=path),
            ]
        )
    return html.Div([])


def complete_root_leaf_path(node: SelectionTree, path: List[str]) -> List[str]:
    """Complete a Dash level change into a full root-leaf path.

    The chosen value, then each deeper level's first child descended to a leaf.

    Args:
        node: The ``(value, label, children)`` subtree the chosen value selects
            at the changed level.
        path: The prefix of chosen values up to and including the changed
            level's value.

    Returns:
        The completed root-leaf path.
    """
    assert isinstance(node, tuple), "Node must be a tuple. node=%r" % (node,)
    assert isinstance(path, list), "Path must be a list. path=%r" % (path,)
    completed_path = list(path)
    _value, _label, children = node
    while children:
        first_child = children[0]
        first_value, _first_label, first_children = first_child
        completed_path.append(first_value)
        children = first_children
    return completed_path
