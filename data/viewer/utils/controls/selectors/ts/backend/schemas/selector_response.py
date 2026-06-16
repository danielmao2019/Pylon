"""Selector cascade response schemas."""

from typing import List

from pydantic import BaseModel


def build_selector_response(option_tree):
    """Build a SelectorResponse from an app's nested option tuple.

    The app owns the tree shape, the lib owns the schema. The supplied tuple is
    the imaginary root of the option tree.

    Args:
        option_tree: The imaginary-root ``(value, label, children)`` tuple where
            ``value`` and ``label`` are strings and ``children`` is a list of
            nested ``(value, label, children)`` tuples.

    Returns:
        A SelectorResponse whose ``root`` is the converted imaginary root.
    """
    return SelectorResponse(root=_to_selection_node(option_tree))


def _to_selection_node(option_tuple):
    """Convert one ``(value, label, children)`` tuple into a SelectionNode.

    Recursion helper: recurses into each child tuple.

    Args:
        option_tuple: A ``(value, label, children)`` tuple where ``value`` and
            ``label`` are strings and ``children`` is a list of nested
            ``(value, label, children)`` tuples.

    Returns:
        A SelectionNode holding its converted children.
    """
    value, label, children = option_tuple
    assert isinstance(value, str), "Option value must be a string. value=%r" % value
    assert isinstance(label, str), "Option label must be a string. label=%r" % label
    assert isinstance(children, list), (
        "Option children must be a list. children=%r" % children
    )
    converted_children = []
    for child_tuple in children:
        converted_children.append(_to_selection_node(child_tuple))
    return SelectionNode(value=value, label=label, children=converted_children)


class SelectorResponse(BaseModel):
    """One selector axis.

    The imaginary root of its option tree, descended recursively along the
    selection path to render the cascade.

    Args:
        root: The imaginary-root node of this axis's option tree.

    Returns:
        Pydantic model for one selector axis response.
    """

    root: "SelectionNode"


class SelectionNode(BaseModel):
    """One option node of a selector axis.

    Holds its value, display label, and child nodes (empty at a leaf), so
    parentage is the nesting itself.

    Args:
        value: This option's selectable value.
        label: This option's display label.
        children: Child option nodes; an empty list marks a leaf.

    Returns:
        Pydantic model for one selector option node.
    """

    value: str
    label: str
    children: List["SelectionNode"]


SelectorResponse.model_rebuild()
