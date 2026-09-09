from typing import List, Optional, Union

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from data.structures.three_d.nerfstudio.convert import _build_colmap_points


def test_a_float_colour_reaches_colmap_on_its_own_range(tmp_path) -> None:
    """A float rgb declares 0 to 1, so it is mapped onto COLMAP's 0-to-255 record rather than cast into it, which would send every mid-grey to black."""
    filepath = tmp_path / "cloud.ply"
    write_ply(filepath=str(filepath), rgb_dtype='f4', rgb_values=[0.0, 0.5, 1.0])

    points = _build_colmap_points(point_cloud_path=filepath)

    colours = [int(points[point_id].rgb[0]) for point_id in range(3)]
    assert (
        colours[0] == 0
    ), f"a float colour of 0.0 is the bottom of COLMAP's range: colours={colours}"
    assert (
        abs(colours[1] - 128) <= 1
    ), f"a float colour of 0.5 is the middle of COLMAP's range: colours={colours}"
    assert (
        colours[2] == 255
    ), f"a float colour of 1.0 is the top of COLMAP's range: colours={colours}"


def test_a_uint16_colour_reaches_colmap_on_its_own_range(tmp_path) -> None:
    """The case the las path produces: a uint16 colour spans 0 to 65535, and casting it to uint8 would wrap rather than scale."""
    filepath = tmp_path / "cloud.ply"
    write_ply(filepath=str(filepath), rgb_dtype='u2', rgb_values=[0, 32768, 65535])

    points = _build_colmap_points(point_cloud_path=filepath)

    colours = [int(points[point_id].rgb[0]) for point_id in range(3)]
    assert (
        colours[0] == 0
    ), f"a uint16 colour of 0 is the bottom of COLMAP's range: colours={colours}"
    assert (
        abs(colours[1] - 128) <= 1
    ), f"a uint16 colour of 32768 is the middle of COLMAP's range: colours={colours}"
    assert (
        colours[2] == 255
    ), f"a uint16 colour of 65535 is the top of COLMAP's range: colours={colours}"


def test_a_double_precision_cloud_narrows_once_and_is_checked(tmp_path) -> None:
    """COLMAP records float32 coordinates, so the narrowing is stated at the load where it is value-checked, not applied silently per point."""
    filepath = tmp_path / "cloud.ply"
    write_ply(filepath=str(filepath), xyz_dtype='f8')

    with pytest.raises(AssertionError):
        _build_colmap_points(point_cloud_path=filepath)


def test_a_cloud_without_colour_is_refused(tmp_path) -> None:
    """A COLMAP point carries a colour, so a cloud that has none is refused rather than exported with a fabricated one."""
    filepath = tmp_path / "cloud.ply"
    write_ply(filepath=str(filepath), with_rgb=False)

    with pytest.raises(AssertionError):
        _build_colmap_points(point_cloud_path=filepath)


def write_ply(
    filepath: str,
    xyz_dtype: str = 'f4',
    rgb_dtype: str = 'u1',
    rgb_values: Optional[List[Union[int, float]]] = None,
    with_rgb: bool = True,
) -> None:
    """Writes a single-element PLY whose coordinate and colour widths the caller names, since this suite's whole subject is which width a column arrives in.

    Args:
        filepath: The destination path of the .ply file to write, as a str.
        xyz_dtype: The ply dtype character the x, y and z columns are stored at, as a str, over coordinates carrying more decimals than float32 holds so an f8 file is one float32 cannot take exactly.
        rgb_dtype: The ply dtype character the red, green and blue columns are stored at, as a str.
        rgb_values: The colour each of the three points carries on every channel, as a list of three numbers on the convention rgb_dtype names, or None to write a default uint8 triple.
        with_rgb: Whether the file carries colour columns at all, as a bool.

    Returns:
        None.
    """
    coordinates = [
        [537000.123456789, 4805000.987654321, 350.192837465],
        [537001.123456789, 4805001.987654321, 351.192837465],
        [537002.123456789, 4805002.987654321, 352.192837465],
    ]
    columns = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    rows = [list(coordinate) for coordinate in coordinates]
    if with_rgb:
        columns = columns + [
            ('red', rgb_dtype),
            ('green', rgb_dtype),
            ('blue', rgb_dtype),
        ]
        values = rgb_values if rgb_values is not None else [0, 128, 255]
        for row, value in zip(rows, values, strict=True):
            row.extend([value, value, value])
    vertices = np.array([tuple(row) for row in rows], dtype=columns)
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
