"""Tests for the ply columns a backend-recoloured segmentation point cloud is written back under."""

from typing import Any, Dict, List, Tuple

import numpy as np
from plyfile import PlyData, PlyElement

from data.viewer.utils.displays.points.ts.backend.apis import (
    _map_segmentation_pc_to_rgb,
)

CLASS_ID_TO_RGB = {0: (10, 20, 30), 1: (200, 210, 220)}


def _write_segmentation_ply(filepath: str, columns: List[Tuple[str, str, Any]]) -> str:
    """Writes a three-point ply carrying exactly the named columns, as the source file a display reads.

    Args:
        filepath: The path of the .ply file to write, as a str.
        columns: The columns to write, each a triple of the column's name as a str, its ply dtype character as a str, and its three values as a sequence.

    Returns:
        The path the ply was written to, as a str.
    """
    vertex_array = np.empty(3, dtype=[(name, char) for name, char, _ in columns])
    for name, _, values in columns:
        vertex_array[name] = values
    PlyData([PlyElement.describe(vertex_array, 'vertex')]).write(filepath)
    return filepath


def _written_columns(filepath: str) -> Dict[str, str]:
    """Reads back the columns a written ply carries beside the ply dtype character each is stored as.

    Args:
        filepath: The path of the .ply file to read, as a str.

    Returns:
        The ply dtype character of each column, keyed by column name, both as str.
    """
    vertex_dtype = PlyData.read(filepath)['vertex'].data.dtype
    return {name: vertex_dtype[name].str.lstrip('<>|') for name in vertex_dtype.names}


def test_a_cloud_with_no_colour_is_written_under_the_ply_column_names(tmp_path):
    """An in-memory field's name stands for its whole block, so both coordinates and colour would write one column against three until the save is told the ply names."""
    filepath = _write_segmentation_ply(
        filepath=str(tmp_path / "segmentation.ply"),
        columns=[
            ('x', 'f4', [0.0, 1.0, 0.0]),
            ('y', 'f4', [0.0, 0.0, 1.0]),
            ('z', 'f4', [0.0, 0.0, 0.0]),
            ('label', 'i4', [0, 1, 1]),
        ],
    )

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    columns = _written_columns(colorized_pc_path)
    assert set(columns.keys()) == {
        'x',
        'y',
        'z',
        'red',
        'green',
        'blue',
    }, f"the recoloured cloud is written under the ply column names: columns={sorted(columns.keys())}"


def test_a_cloud_that_already_had_colour_is_recoloured_on_the_class_map_s_own_range(
    tmp_path,
):
    """The class colours are 0 to 255 whatever the source file declared its own colour to be, and a fresh cloud is what keeps the source's convention from being claimed over them."""
    filepath = _write_segmentation_ply(
        filepath=str(tmp_path / "segmentation.ply"),
        columns=[
            ('x', 'f4', [0.0, 1.0, 0.0]),
            ('y', 'f4', [0.0, 0.0, 1.0]),
            ('z', 'f4', [0.0, 0.0, 0.0]),
            ('label', 'i4', [0, 1, 1]),
            ('red', 'u2', [0, 32767, 65535]),
            ('green', 'u2', [0, 32767, 65535]),
            ('blue', 'u2', [0, 32767, 65535]),
        ],
    )

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    columns = _written_columns(colorized_pc_path)
    assert all(
        columns[name] == 'u1' for name in ('red', 'green', 'blue')
    ), f"the class colours are written on their own uint8 range rather than the source's: columns={columns}"
    written = PlyData.read(colorized_pc_path)['vertex'].data
    np.testing.assert_array_equal(
        np.stack([written['red'], written['green'], written['blue']], axis=1),
        np.array([[10, 20, 30], [200, 210, 220], [200, 210, 220]], dtype=np.uint8),
    )


def test_the_written_cloud_carries_the_class_colours_and_nothing_else(tmp_path):
    """The resource exists to be displayed, so the source's remaining columns reach no output column."""
    filepath = _write_segmentation_ply(
        filepath=str(tmp_path / "segmentation.ply"),
        columns=[
            ('x', 'f4', [0.0, 1.0, 0.0]),
            ('y', 'f4', [0.0, 0.0, 1.0]),
            ('z', 'f4', [0.0, 0.0, 0.0]),
            ('label', 'i4', [0, 1, 1]),
            ('intensity', 'f4', [0.5, 0.6, 0.7]),
        ],
    )

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    columns = _written_columns(colorized_pc_path)
    assert (
        'intensity' not in columns
    ), f"a column the display does not read reaches no output column: columns={sorted(columns.keys())}"
