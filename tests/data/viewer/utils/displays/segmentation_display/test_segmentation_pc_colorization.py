"""Tests for the ply columns a backend-recoloured segmentation point cloud is written back under."""

import numpy as np
from plyfile import PlyData, PlyElement

from data.viewer.utils.displays.points.ts.backend.apis import (
    _map_segmentation_pc_to_rgb,
)

CLASS_ID_TO_RGB = {0: (10, 20, 30), 1: (200, 210, 220)}


def test_a_cloud_with_no_colour_is_written_under_the_ply_column_names(tmp_path):
    """An in-memory field's name stands for its whole block, so both coordinates and colour would write one column against three until the save is told the ply names."""
    filepath = str(tmp_path / "segmentation.ply")
    vertex_array = np.empty(
        3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'i4')]
    )
    vertex_array['x'] = [0.0, 1.0, 0.0]
    vertex_array['y'] = [0.0, 0.0, 1.0]
    vertex_array['z'] = [0.0, 0.0, 0.0]
    vertex_array['label'] = [0, 1, 1]
    PlyData([PlyElement.describe(vertex_array, 'vertex')]).write(filepath)

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    vertex_dtype = PlyData.read(colorized_pc_path)['vertex'].data.dtype
    assert set(vertex_dtype.names) == {
        'x',
        'y',
        'z',
        'red',
        'green',
        'blue',
    }, f"the recoloured cloud is written under the ply column names: columns={sorted(vertex_dtype.names)}"


def test_a_cloud_that_already_had_colour_is_recoloured_on_the_class_map_s_own_range(
    tmp_path,
):
    """The class colours are 0 to 255 whatever the source file declared its own colour to be, and a fresh cloud is what keeps the source's convention from being claimed over them."""
    filepath = str(tmp_path / "segmentation.ply")
    vertex_array = np.empty(
        3,
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('label', 'i4'),
            ('red', 'u2'),
            ('green', 'u2'),
            ('blue', 'u2'),
        ],
    )
    vertex_array['x'] = [0.0, 1.0, 0.0]
    vertex_array['y'] = [0.0, 0.0, 1.0]
    vertex_array['z'] = [0.0, 0.0, 0.0]
    vertex_array['label'] = [0, 1, 1]
    vertex_array['red'] = [0, 32767, 65535]
    vertex_array['green'] = [0, 32767, 65535]
    vertex_array['blue'] = [0, 32767, 65535]
    PlyData([PlyElement.describe(vertex_array, 'vertex')]).write(filepath)

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    written = PlyData.read(colorized_pc_path)['vertex'].data
    assert all(
        written.dtype[name].str.lstrip('<>|') == 'u1'
        for name in ('red', 'green', 'blue')
    ), f"the class colours are written on their own uint8 range rather than the source's: vertex_dtype={written.dtype}"
    np.testing.assert_array_equal(
        np.stack([written['red'], written['green'], written['blue']], axis=1),
        np.array([[10, 20, 30], [200, 210, 220], [200, 210, 220]], dtype=np.uint8),
    )


def test_the_written_cloud_carries_the_class_colours_and_nothing_else(tmp_path):
    """The resource exists to be displayed, so the source's remaining columns reach no output column."""
    filepath = str(tmp_path / "segmentation.ply")
    vertex_array = np.empty(
        3,
        dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('label', 'i4'),
            ('intensity', 'f4'),
        ],
    )
    vertex_array['x'] = [0.0, 1.0, 0.0]
    vertex_array['y'] = [0.0, 0.0, 1.0]
    vertex_array['z'] = [0.0, 0.0, 0.0]
    vertex_array['label'] = [0, 1, 1]
    vertex_array['intensity'] = [0.5, 0.6, 0.7]
    PlyData([PlyElement.describe(vertex_array, 'vertex')]).write(filepath)

    colorized_pc_path = _map_segmentation_pc_to_rgb(
        segmentation_pc_path=filepath, class_id_to_rgb=CLASS_ID_TO_RGB
    )

    vertex_dtype = PlyData.read(colorized_pc_path)['vertex'].data.dtype
    assert (
        'intensity' not in vertex_dtype.names
    ), f"a column the display does not read reaches no output column: columns={sorted(vertex_dtype.names)}"
