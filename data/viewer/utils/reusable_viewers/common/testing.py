"""Shared test helpers for Dash viewer structure."""

from typing import Any, List, Set

from dash.development.base_component import Component


def collect_component_ids(component: Any) -> Set[str]:
    """Collect component ids from a Dash layout tree.

    Args:
        component: Dash component or nested component collection.

    Returns:
        Set of string component ids.
    """
    ids = set()
    if component is None:
        return ids
    if isinstance(component, (list, tuple)):
        for child in component:
            ids.update(collect_component_ids(child))
        return ids
    if isinstance(component, Component):
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str):
            ids.add(component_id)
        children = getattr(component, "children", None)
        ids.update(collect_component_ids(children))
    return ids


def assert_callback_ids_exist(app_layout: Any, callback_map: dict) -> None:
    """Assert callback dependencies are declared in the app layout.

    Args:
        app_layout: Dash app layout.
        callback_map: Dash callback map.

    Returns:
        None.
    """
    component_ids = collect_component_ids(app_layout)
    assert component_ids, "Layout must contain component ids. component_ids=%r" % (
        component_ids,
    )
    for callback_id, callback_spec in callback_map.items():
        for role in ["inputs", "state"]:
            dependencies = callback_spec[role]
            assert isinstance(dependencies, list), (
                "Callback dependencies must be a list. callback_id=%s role=%s value=%r"
                % (callback_id, role, dependencies)
            )
            for dependency in dependencies:
                dependency_id = dependency["id"]
                assert dependency_id in component_ids, (
                    "Callback dependency id is missing from layout. callback_id=%s id=%s"
                    " known_ids=%s"
                    % (callback_id, dependency_id, sorted(component_ids))
                )
        outputs = callback_spec["output"]
        output_ids = _output_ids(outputs)
        for output_id in output_ids:
            assert output_id in component_ids, (
                "Callback output id is missing from layout. callback_id=%s id=%s"
                " known_ids=%s"
                % (callback_id, output_id, sorted(component_ids))
            )


def assert_callbacks_have_one_input(callback_map: dict) -> None:
    """Assert each registered callback has exactly one Input.

    Args:
        callback_map: Dash callback map.

    Returns:
        None.
    """
    assert isinstance(callback_map, dict), "Callback map must be a dict. value=%r" % (
        callback_map,
    )
    for callback_id, callback_spec in callback_map.items():
        inputs = callback_spec["inputs"]
        assert len(inputs) == 1, (
            "Callback must have exactly one Input. callback_id=%s inputs=%r"
            % (callback_id, inputs)
        )


def _output_ids(outputs: Any) -> List[str]:
    """Extract string ids from Dash callback output objects.

    Args:
        outputs: Dash output specification.

    Returns:
        Output component ids.
    """
    if isinstance(outputs, list):
        ids = []
        for output in outputs:
            ids.extend(_output_ids(output))
        return ids
    component_id = getattr(outputs, "component_id", None)
    if isinstance(component_id, str):
        return [component_id]
    output_id = str(outputs).split(".", maxsplit=1)[0]
    return [output_id]
