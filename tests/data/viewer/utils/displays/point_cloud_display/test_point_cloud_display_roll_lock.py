"""Tests that `create_point_cloud_display` renders the roll lock its caller asked for.

The axis a caller names through `lock_roll` is inert unless it reaches the layout of
the figure the factory actually returns, so every assertion here reads the rendered
`figure.layout.scene` rather than the factory's inputs. The no-axis default is the
other half: it must leave the rendered figure exactly as it was before the parameter
existed, which the assertions below read as the scene carrying no camera
configuration at all.
"""

import math
from typing import Dict, Tuple

import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.displays.points.dash.core_points_display import (
    create_point_cloud_display,
)

# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.0, 0.95)


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


def test_a_supplied_axis_reaches_the_rendered_scene(point_cloud):
    """A display given a lock_roll renders the roll-locking dragmode with the caller's axis as the camera up vector."""
    figure = create_point_cloud_display(
        pc=point_cloud,
        title="Point Cloud",
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
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


def test_no_axis_renders_no_camera_configuration(point_cloud):
    """A display given no lock_roll renders the scene it rendered before the parameter existed, which carries no dragmode and no camera up vector."""
    figure = create_point_cloud_display(pc=point_cloud, title="Point Cloud")

    scene = figure.layout.scene
    assert scene.dragmode is None, (
        "A display given no roll-lock axis must not write a dragmode into the "
        f"rendered scene, which would change what every existing caller renders. {scene.dragmode=}"
    )
    assert (scene.camera.up.x, scene.camera.up.y, scene.camera.up.z) == (
        None,
        None,
        None,
    ), (
        "A display given no roll-lock axis must not write a camera up vector into "
        f"the rendered scene. {scene.camera.up=}"
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
