"""Tests for the colour convention the Dash point-cloud scene reads its per-point colours on."""

import numpy as np
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.displays.points.dash.core_points_display import (
    create_dash_points_scene,
)

XYZ = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


def test_a_uint16_colour_is_read_over_its_own_range():
    """The convention is read off what the field MEANS, not off the tensor torch had to widen it into, which is the defect this design exists to close."""
    pc = PointCloud(
        xyz=XYZ,
        data={
            'rgb': torch.tensor(
                [[0, 0, 0], [32767, 32767, 32767], [65535, 65535, 65535]],
                dtype=torch.int32,
            )
        },
        meta_data={
            'xyz': {'dtype': 'float32', 'layout': ('xyz',)},
            'rgb': {'dtype': 'uint16', 'layout': ('rgb',)},
        },
    )
    assert (
        pc.rgb.dtype == torch.int32
    ), f"a uint16 colour is parked in an int32 tensor: pc.rgb.dtype={pc.rgb.dtype}"

    trace = create_dash_points_scene(point_cloud=pc)

    np.testing.assert_array_equal(
        trace.marker.color,
        np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255]], dtype=np.float64),
    )


def test_a_float_colour_is_read_as_zero_to_one():
    """A float dtype declares the 0-to-1 convention, so no rescale is guessed from the values."""
    pc = PointCloud(
        xyz=XYZ,
        data={
            'rgb': torch.tensor(
                [[0.0, 0.0, 0.0], [0.2, 0.4, 0.6], [1.0, 1.0, 1.0]],
                dtype=torch.float32,
            )
        },
    )
    assert (
        pc.meta_data['rgb']['dtype'] == 'float32'
    ), f"the record names the float convention: rgb entry={pc.meta_data['rgb']}"

    trace = create_dash_points_scene(point_cloud=pc)

    np.testing.assert_array_equal(
        trace.marker.color,
        np.array([[0, 0, 0], [51, 102, 153], [255, 255, 255]], dtype=np.float64),
    )


def test_a_uint8_colour_is_passed_through():
    """uint8 already spans the display's own range, so the conversion is the identity rather than a second rescale."""
    rgb = torch.tensor([[0, 10, 20], [30, 40, 50], [253, 254, 255]], dtype=torch.uint8)
    pc = PointCloud(xyz=XYZ, data={'rgb': rgb})
    assert (
        pc.meta_data['rgb']['dtype'] == 'uint8'
    ), f"the record names the uint8 convention: rgb entry={pc.meta_data['rgb']}"

    trace = create_dash_points_scene(point_cloud=pc)

    np.testing.assert_array_equal(trace.marker.color, rgb.numpy().astype(np.float64))
