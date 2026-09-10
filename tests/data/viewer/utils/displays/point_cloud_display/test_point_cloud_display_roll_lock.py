"""Tests that `create_point_cloud_display` renders the roll lock its caller asked for.

The axis a caller names through `lock_roll` is inert unless it reaches the layout of
the figure the factory actually returns, so every assertion here reads the rendered
`figure.layout.scene` rather than the factory's inputs. The no-axis default is the
other half: it must render the free-roll dragmode and pin no axis, which the
assertions below read as changing exactly one key -- the dragmode -- of the figure
rendered before any camera-control configuration is merged into its scene.

A lock lives on the Dash app rather than on a figure, so the axis reaching the rendered scene is only the framing the lock starts from. The clauses below therefore also read the app the factory was handed, which is where a caller who named an axis and nothing else would otherwise be left with a panel that only looks locked.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go
import pytest
import torch
from dash import Dash

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    PLOTLY_FREE_ROLL_DRAGMODE,
    PLOTLY_POSE_CLAMPING_DRAGMODE,
)
from data.viewer.utils.displays.points.dash.core_points_display import (
    _create_point_cloud_figure,
    create_point_cloud_display,
)

# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.0, 0.95)
# Component id of the graph the roll-locked figure below is mounted under, and
# therefore the id the registration the factory performs must address.
LOCKED_GRAPH_ID = "locked-display-graph"
# The rejection each missing half of the lock target must be named by. Matching the
# factory's own wording rather than the word `app` or `graph_id` alone is what keeps
# these clauses from passing on the registration helper's type assertions, which
# report a wrong type instead of naming what the lock is missing.
MISSING_APP_MESSAGE = re.escape("Expected an `app` alongside `lock_roll`")
MISSING_GRAPH_ID_MESSAGE = re.escape("Expected a `graph_id` alongside `lock_roll`")
# The rejection an `app` handed in with no axis must be named by. Matching this
# contract's own wording rather than the word `app` or `lock_roll` alone is what keeps
# the clause from passing on some unrelated later failure that happens to mention one
# of those words instead of naming what the lock is missing.
MISSING_LOCK_ROLL_MESSAGE = re.escape("Expected a `lock_roll` alongside `app`")
# The rejection a `graph_id` handed in with no axis must be named by. The sibling
# component factories accept one, because there the id names the `dcc.Graph` they
# return; a figure carries no id, so here the id is read for nothing but the lock and
# this contract's own wording is what the clause matches.
MISSING_LOCK_ROLL_FOR_GRAPH_ID_MESSAGE = re.escape(
    "Expected a `lock_roll` alongside `graph_id`"
)
# The rendered JSON path the no-axis default is allowed to change, and the only one.
DRAGMODE_JSON_PATH = "layout.scene.dragmode"
# Call-site shapes whose rendered figures each carry scene content the merged camera
# controls could silently discard: the caller's camera, the fixed axis ranges, the
# highlight trace, and the marker styling.
NO_AXIS_CALL_SHAPES: Dict[str, Dict[str, Any]] = {
    "minimal": {},
    "camera_state": {
        "camera_state": {
            "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    },
    "axis_ranges": {
        "axis_ranges": {"x": (-1.0, 4.0), "y": (-1.0, 3.0), "z": (-1.0, 1.0)}
    },
    "highlight_indices": {"highlight_indices": torch.tensor([0, 2])},
    "point_style": {"point_size": 5.0, "point_opacity": 0.25},
}


@pytest.fixture
def point_cloud():
    """Fixture providing a small colored point cloud."""
    return PointCloud(
        xyz=torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
        data={
            'rgb': torch.tensor(
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                dtype=torch.uint8,
            )
        },
    )


def clientside_registrations(app: Dash) -> List[Dict[str, Any]]:
    """Collect the clientside callbacks registered on a Dash app.

    Args:
        app: Dash app whose callback registrations are read.

    Returns:
        List of the app's callback registration dicts whose
        `clientside_function` is set, each carrying an `inputs` list of
        `{"id", "property"}` dicts.
    """
    return [
        registration
        for registration in app._callback_list
        if registration["clientside_function"] is not None
    ]


def expected_camera_up(lock_roll: Tuple[float, float, float]) -> Dict[str, float]:
    """Compute the unit-length Plotly camera up vector a lock axis must produce.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        Dict with `x`, `y`, and `z` unit-length components.
    """
    length = math.sqrt(sum(component * component for component in lock_roll))
    return {
        "x": lock_roll[0] / length,
        "y": lock_roll[1] / length,
        "z": lock_roll[2] / length,
    }


def render_before_camera_controls(
    pc: PointCloud,
    title: str,
    color_key: Optional[str] = None,
    color_type: Optional[str] = None,
    highlight_indices: Optional[torch.Tensor] = None,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> go.Figure:
    """Render the figure the display builds before any camera-control configuration is merged into its scene.

    Args:
        pc: Point cloud to render, whose `xyz` field holds positions of shape [N, 3] and whose `rgb` field holds per-point colors.
        title: Title the rendered figure carries.
        color_key: Optional field name to read per-point labels from.
        color_type: Optional label rendering, either `"classification"` or `"regression"`.
        highlight_indices: Optional int64 tensor of shape [K] indexing the points drawn as a highlight trace.
        point_size: Marker size of the rendered points.
        point_opacity: Marker opacity of the rendered points, in [0, 1].
        camera_state: Optional Plotly gl3d camera dict placed into `layout.scene.camera`.
        axis_ranges: Optional fixed per-axis `(low, high)` ranges keyed by `"x"`, `"y"`, and `"z"`.

    Returns:
        Plotly figure whose `layout.scene` carries only what the point-cloud data, `camera_state`, and `axis_ranges` put there.
    """
    figure = _create_point_cloud_figure(
        pc=pc,
        color_key=color_key,
        color_type=color_type,
        highlight_indices=highlight_indices,
        point_size=point_size,
        point_opacity=point_opacity,
        axis_ranges=axis_ranges,
        camera_state=camera_state,
    )
    figure.update_layout(title=title, uirevision='camera')
    return figure


def flatten_json_paths(value: Any, prefix: str, flattened: Dict[str, Any]) -> None:
    """Flatten a parsed figure JSON into a mapping from dotted leaf path to leaf value.

    Args:
        value: Parsed JSON value to flatten, either a dict, a list, or a scalar leaf.
        prefix: Dotted path already accumulated for `value`, empty at the root.
        flattened: Mapping the leaf paths and their values are written into.

    Returns:
        None.
    """
    if isinstance(value, dict):
        for key in value:
            flatten_json_paths(
                value=value[key],
                prefix=f"{prefix}.{key}" if prefix else key,
                flattened=flattened,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten_json_paths(
                value=item, prefix=f"{prefix}[{index}]", flattened=flattened
            )
    else:
        flattened[prefix] = value


def differing_json_paths(before: Dict[str, Any], after: Dict[str, Any]) -> Set[str]:
    """Compute the leaf paths at which two parsed figure JSONs disagree.

    Args:
        before: Parsed `figure.to_json()` of the figure rendered first.
        after: Parsed `figure.to_json()` of the figure rendered second.

    Returns:
        Set of dotted leaf paths whose values differ, counting a path present in only one side as differing.
    """
    flattened_before: Dict[str, Any] = {}
    flattened_after: Dict[str, Any] = {}
    flatten_json_paths(value=before, prefix="", flattened=flattened_before)
    flatten_json_paths(value=after, prefix="", flattened=flattened_after)
    absent = object()
    return {
        path
        for path in set(flattened_before) | set(flattened_after)
        if flattened_before.get(path, absent) != flattened_after.get(path, absent)
    }


def test_a_supplied_axis_reaches_the_rendered_scene(point_cloud):
    """A display given a lock_roll renders the roll-locking dragmode with the caller's axis as the camera up vector."""
    figure = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    )

    scene = figure.layout.scene
    assert scene.dragmode == "orbit", (
        "A display given a roll-lock axis must render the Plotly dragmode that "
        f"carries that axis through re-render. {scene.dragmode=}"
    )
    up = expected_camera_up(NON_AXIS_ALIGNED_LOCK_ROLL)
    assert (scene.camera.up.x, scene.camera.up.y, scene.camera.up.z) == (
        up["x"],
        up["y"],
        up["z"],
    ), (
        "A display given a roll-lock axis must carry that axis into the rendered "
        f"camera up vector. {scene.camera.up=} {up=}"
    )


def test_an_axis_alone_locks_the_display(point_cloud):
    """A figure built with an axis registers the roll lock itself, so a caller that names the axis is not left with an unlocked panel for want of a second call it had to know about."""
    app = Dash(__name__)

    create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=app,
        graph_id=LOCKED_GRAPH_ID,
    )

    registrations = clientside_registrations(app=app)
    assert len(registrations) == 1, (
        "A figure built with a roll-lock axis must put the lock on the app itself, "
        "since seeding the rendered camera up vector holds roll through re-render "
        f"but never through a drag. {registrations=}"
    )
    assert registrations[0]["inputs"] == [
        {"id": LOCKED_GRAPH_ID, "property": "relayoutData"}
    ], (
        "The lock the factory registers must be driven by the graph the figure is "
        "mounted under, which is how every camera change on it reaches the lock. "
        f"{registrations[0]=} {LOCKED_GRAPH_ID=}"
    )


def test_an_axis_without_the_app_is_rejected(point_cloud):
    """Naming an axis with no app to register the lock on is rejected, rather than silently returning a figure that only looks locked."""
    with pytest.raises(AssertionError, match=MISSING_APP_MESSAGE):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            graph_id=LOCKED_GRAPH_ID,
        )


def test_an_axis_without_the_graph_id_is_rejected(point_cloud):
    """Naming an axis with no graph id for the lock to address is rejected, rather than silently returning a figure that only looks locked."""
    with pytest.raises(AssertionError, match=MISSING_GRAPH_ID_MESSAGE):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            app=Dash(__name__),
        )


def test_an_app_without_the_axis_is_rejected(point_cloud):
    """Handing in an app with no axis for it to lock about is rejected, rather than silently returning the unlocked figure a caller who believed they were locking would otherwise get."""
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            app=Dash(__name__),
        )
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            app=Dash(__name__),
            graph_id=LOCKED_GRAPH_ID,
        )


def test_a_graph_id_without_the_axis_is_rejected(point_cloud):
    """Handing in a graph id with no axis for the lock to address is rejected, because a figure carries no id of its own, so unlike the sibling component factories there is no graph here for a lone id to name."""
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_FOR_GRAPH_ID_MESSAGE):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            graph_id=LOCKED_GRAPH_ID,
        )


def test_naming_none_of_the_three_changes_only_the_dragmode(point_cloud):
    """Naming the axis, the app, and the graph id all as None renders the pre-camera-control figure changed at the dragmode alone, which is the figure the unlocked path rendered before a lock target existed, so no existing call site moves."""
    before = json.loads(
        render_before_camera_controls(pc=point_cloud, title="Point Cloud").to_json()
    )
    after = json.loads(
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            lock_roll=None,
            app=None,
            graph_id=None,
        ).to_json()
    )

    assert differing_json_paths(before=before, after=after) == {DRAGMODE_JSON_PATH}, (
        "Naming no lock target must leave the rendered figure the unlocked path "
        "always rendered, changed at the dragmode alone, so neither the app nor "
        "the graph id reaches the figure. "
        f"{differing_json_paths(before=before, after=after)=}"
    )


def test_no_axis_renders_the_free_roll_dragmode(point_cloud):
    """A display given no lock_roll renders the dragmode that leaves roll free, never the one that pins camera up to world +Z, and pins no axis of its own."""
    figure = create_point_cloud_display(pc=point_cloud, title="Point Cloud")

    scene = figure.layout.scene
    assert scene.dragmode == PLOTLY_FREE_ROLL_DRAGMODE, (
        "A display given no roll-lock axis must render the free-roll dragmode, "
        "because a scene carrying no dragmode leaves Plotly's own gl3d default in "
        f"force. {scene.dragmode=} {PLOTLY_FREE_ROLL_DRAGMODE=}"
    )
    assert scene.dragmode != PLOTLY_POSE_CLAMPING_DRAGMODE, (
        "A display given no roll-lock axis must never render the dragmode that "
        "clamps camera up onto world +Z, which is a roll lock about an axis the "
        f"caller never named. {scene.dragmode=} {PLOTLY_POSE_CLAMPING_DRAGMODE=}"
    )
    assert (scene.camera.up.x, scene.camera.up.y, scene.camera.up.z) == (
        None,
        None,
        None,
    ), (
        "A display given no roll-lock axis must not write a camera up vector into "
        f"the rendered scene. {scene.camera.up=}"
    )


@pytest.mark.parametrize(
    "call_kwargs",
    list(NO_AXIS_CALL_SHAPES.values()),
    ids=list(NO_AXIS_CALL_SHAPES.keys()),
)
def test_no_axis_changes_only_the_dragmode(point_cloud, call_kwargs):
    """A display given no lock_roll renders the pre-camera-control figure with exactly one key changed: the dragmode, which the merge adds as the free-roll one."""
    before = json.loads(
        render_before_camera_controls(
            pc=point_cloud, title="Point Cloud", **call_kwargs
        ).to_json()
    )
    after = json.loads(
        create_point_cloud_display(
            pc=point_cloud, title="Point Cloud", **call_kwargs
        ).to_json()
    )

    assert differing_json_paths(before=before, after=after) == {DRAGMODE_JSON_PATH}, (
        "Merging the no-axis camera controls must change the rendered figure at the "
        "dragmode alone, so no axis range, camera, trace, or uirevision the scene "
        "already carried is discarded. "
        f"{differing_json_paths(before=before, after=after)=} {call_kwargs=}"
    )
    flattened_before: Dict[str, Any] = {}
    flatten_json_paths(value=before, prefix="", flattened=flattened_before)
    assert DRAGMODE_JSON_PATH not in flattened_before, (
        "The figure rendered before the camera controls are merged must carry no "
        f"dragmode. {flattened_before[DRAGMODE_JSON_PATH]=}"
    )
    flattened_after: Dict[str, Any] = {}
    flatten_json_paths(value=after, prefix="", flattened=flattened_after)
    assert flattened_after[DRAGMODE_JSON_PATH] == PLOTLY_FREE_ROLL_DRAGMODE, (
        "The dragmode the no-axis merge adds must be the free-roll one. "
        f"{flattened_after[DRAGMODE_JSON_PATH]=} {PLOTLY_FREE_ROLL_DRAGMODE=}"
    )


def test_the_default_is_inert(point_cloud):
    """Passing lock_roll=None renders byte-identically to not passing lock_roll at all."""
    omitted = create_point_cloud_display(pc=point_cloud, title="Point Cloud")
    explicit_none = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        lock_roll=None,
    )

    assert omitted.to_json() == explicit_none.to_json(), (
        "The lock_roll default must render exactly what omitting the argument "
        f"renders. {omitted.to_json()=} {explicit_none.to_json()=}"
    )


def test_the_two_roll_settings_render_different_scenes(point_cloud):
    """The rendered scene differs between the locked and unlocked settings, so the axis cannot be silently discarded."""
    free_scene = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
    ).layout.scene
    locked_scene = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    ).layout.scene

    assert free_scene != locked_scene, (
        "Naming a roll-lock axis must change the rendered scene. "
        f"{free_scene=} {locked_scene=}"
    )


def test_a_supplied_axis_composes_with_camera_state(point_cloud):
    """A lock axis adds the camera up vector to the caller's camera_state rather than replacing it."""
    camera_state = {
        "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    }

    camera = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        camera_state=camera_state,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    ).layout.scene.camera

    assert (camera.eye.x, camera.eye.y, camera.eye.z) == (1.25, 1.25, 1.25), (
        "A roll-lock axis must leave the caller's camera_state eye in place. "
        f"{camera.eye=} {camera_state=}"
    )
    up = expected_camera_up(NON_AXIS_ALIGNED_LOCK_ROLL)
    assert (camera.up.x, camera.up.y, camera.up.z) == (up["x"], up["y"], up["z"]), (
        "A roll-lock axis must reach the camera up vector even when the caller "
        f"also supplied a camera_state. {camera.up=} {up=}"
    )


@pytest.mark.parametrize(
    "lock_roll",
    [
        [0.3, 0.0, 0.95],
        (0.3, 0.95),
        (0.3, 0.0, 0.0, 0.95),
        (0, 0, 1),
        (0.0, 0.0, 0.0),
    ],
    ids=[
        "list_not_tuple",
        "two_components",
        "four_components",
        "int_components",
        "all_zero",
    ],
)
def test_invalid_lock_roll_rejected(point_cloud, lock_roll):
    """A lock_roll that is not a non-zero 3-tuple of floats is rejected before anything is rendered."""
    with pytest.raises(AssertionError, match="lock_roll"):
        create_point_cloud_display(
            pc=point_cloud,
            title="Point Cloud",
            lock_roll=lock_roll,
        )
