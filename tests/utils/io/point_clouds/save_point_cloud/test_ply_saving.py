import tempfile

import laspy
import numpy as np
import open3d as o3d
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
from data.structures.three_d.point_cloud.point_cloud import PointCloud


@pytest.fixture
def pc() -> PointCloud:
    """Build the in-memory cloud every case below saves, handed in as the three separately named columns a PLY file holds rather than as one coordinate block.

    Args:
        None.

    Returns:
        A PointCloud of eight points on the cpu device, whose record maps xyz back onto the three columns 'x', 'y' and 'z' a PLY save writes under.
    """
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    columns = {
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'z': coordinates[:, 2],
    }
    return PointCloud(data=columns, device='cpu')


def write_las(filepath: str, num_points: int = 8, with_rgb: bool = False) -> np.ndarray:
    """Write a LAS carrying xyz plus, on request, the uint16 colour dimensions this suite saves back out.

    Args:
        filepath: Destination path of the .las file.
        num_points: Number of points the file carries, as an int.
        with_rgb: Whether the file carries the red, green and blue dimensions, as a bool.

    Returns:
        The uint16 colours written, as an [N, 3] numpy array of dtype uint16, or an empty [N, 0] uint16 array when with_rgb is False.
    """
    header = laspy.LasHeader(point_format=2 if with_rgb else 0, version='1.2')
    las_data = laspy.LasData(header)
    las_data.x = np.arange(num_points, dtype=np.float64)
    las_data.y = np.arange(num_points, dtype=np.float64) * 2.0
    las_data.z = np.arange(num_points, dtype=np.float64) * 3.0
    colors = np.zeros((num_points, 0), dtype=np.uint16)
    if with_rgb:
        # whole multiples of 257, so the 0-to-65535 convention maps onto the 0-to-255 one exactly
        colors = np.tile(
            np.array([128 * 257, 64 * 257, 192 * 257], dtype=np.uint16),
            (num_points, 1),
        )
        las_data.red = colors[:, 0]
        las_data.green = colors[:, 1]
        las_data.blue = colors[:, 2]
    las_data.write(filepath)
    return colors


def write_pcd(
    filepath: str, num_points: int = 8, with_colors: bool = False
) -> np.ndarray:
    """Write a PCD through Open3D's tensor IO, so this suite can save a cloud whose fields each carry one attribute name.

    Args:
        filepath: Destination path of the .pcd file.
        num_points: Number of points the file carries, as an int.
        with_colors: Whether the file carries the colors attribute, as a bool.

    Returns:
        The positions written, as an [N, 3] numpy array of dtype float32.
    """
    positions = np.arange(num_points * 3, dtype=np.float32).reshape(num_points, 3)
    tensor_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(positions))
    if with_colors:
        tensor_pcd.point.colors = o3d.core.Tensor(
            np.full((num_points, 3), 0.5, dtype=np.float32)
        )
    o3d.t.io.write_point_cloud(filepath, tensor_pcd)
    return positions


def test_basic_ply_saving(pc):
    """Coordinates written to a PLY under the column names their meta data entry gives them come back as the ones that were saved."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = pc.xyz.cpu().numpy().copy()

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.xyz.cpu().numpy(), coordinates)


def test_numpy_array_input(pc):
    """A PointCloud built from an np.array saves on the same terms as one built from a tensor."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = pc.xyz.cpu().numpy().copy()

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.xyz.cpu().numpy(), coordinates)


def test_large_coordinates_precision():
    """UTM-magnitude coordinates survive the round trip to within float32's own resolution."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.stack(
        [
            537000.0 + np.arange(8, dtype=np.float32),
            4805000.0 + np.arange(8, dtype=np.float32),
            350.0 + np.arange(8, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    error = np.abs(loaded.xyz.cpu().numpy() - coordinates).max()
    assert (
        error <= torch.finfo(torch.float32).eps * np.abs(coordinates).max()
    ), f"coordinates come back within float32's own resolution of the magnitude: error={error}, magnitude={np.abs(coordinates).max()}, eps={torch.finfo(torch.float32).eps}"


def test_a_float_rgb_is_written_on_the_integer_range_its_meta_data_names():
    """A float rgb declares 0 to 1 and a uint8 target declares 0 to 255, so save converts between the two conventions the dtypes name."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    # the 0-to-1 counterparts of whole 0-to-255 steps
    colors = np.tile(np.array([0.0, 128 / 255, 1.0], dtype=np.float64), (8, 1))
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    save_point_cloud(
        pc,
        filepath,
        meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['red'], np.zeros(8, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['green'], np.full(8, 128, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['blue'], np.full(8, 255, dtype=np.uint8))


def test_an_integer_rgb_is_written_on_its_own_range_untouched():
    """A uint8 rgb saved under a uint8 target is already on the target range, so no conversion happens at all."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    colors = np.tile(np.array([10, 128, 250], dtype=np.uint8), (8, 1))
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    save_point_cloud(
        pc,
        filepath,
        meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['red'], colors[:, 0])
    np.testing.assert_array_equal(vertex_data['green'], colors[:, 1])
    np.testing.assert_array_equal(vertex_data['blue'], colors[:, 2])


def test_a_las_loaded_colour_round_trips_to_ply_with_nothing_supplied():
    """A uint16 colour loaded from las sits in an int32 tensor, and saving with no meta at all writes it back as u2 under the file's own column names."""
    with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp_file:
        las_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    colors = write_las(las_path, with_rgb=True)

    loaded = load_point_cloud(filepath=las_path, device='cpu')

    assert (
        loaded.rgb.dtype == torch.int32
    ), f"a uint16 colour is parked in an int32 tensor: rgb.dtype={loaded.rgb.dtype}"

    save_point_cloud(loaded, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    for column_name in ('red', 'green', 'blue'):
        assert vertex_data.dtype[column_name] == np.dtype(
            'u2'
        ), f"a uint16 colour reaches a u2 column: column_name={column_name}, stored dtype={vertex_data.dtype[column_name]}"
    np.testing.assert_array_equal(vertex_data['red'], colors[:, 0])
    np.testing.assert_array_equal(vertex_data['green'], colors[:, 1])
    np.testing.assert_array_equal(vertex_data['blue'], colors[:, 2])


def test_a_las_loaded_colour_saves_through_its_own_range_with_no_dtype_stated():
    """A uint16 colour sits in an int32 tensor, and save reads the range off what the field means rather than off the tensor it is parked in."""
    with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp_file:
        las_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    colors = write_las(las_path, with_rgb=True)

    loaded = load_point_cloud(filepath=las_path, device='cpu')
    save_point_cloud(loaded, filepath, meta_data={'rgb': {'dtype': 'uint8'}})
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['red'], colors[:, 0] // 257)
    np.testing.assert_array_equal(vertex_data['green'], colors[:, 1] // 257)
    np.testing.assert_array_equal(vertex_data['blue'], colors[:, 2] // 257)


def test_a_uint16_las_colour_round_trips_through_its_own_range():
    """A uint16 colour spans 0 to 65535, and converting it to a uint8 target uses that range rather than a guessed one."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    # whole multiples of 257, and so exactly recoverable
    colors = np.tile(np.array([0, 32896, 65535], dtype=np.uint16), (8, 1))
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    save_point_cloud(
        pc,
        filepath,
        meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['red'], np.zeros(8, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['green'], np.full(8, 128, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['blue'], np.full(8, 255, dtype=np.uint8))


def test_a_signed_colour_saves_through_its_own_range():
    """An int8 colour spans that dtype's own range, so it maps onto a u1 column by the offset the range mapping gives rather than being read as unsigned."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    colors = np.tile(np.array([-128, 0, 127], dtype=np.int8), (8, 1))
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    save_point_cloud(
        pc,
        filepath,
        meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['red'], np.zeros(8, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['green'], np.full(8, 128, dtype=np.uint8))
    np.testing.assert_array_equal(vertex_data['blue'], np.full(8, 255, dtype=np.uint8))


def test_a_colour_off_the_target_grid_is_refused_at_save():
    """A colour that rounds into the target grid would return as a different colour than the one saved, and save refuses rather than writing it."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    # 1 converts to 0 and back to 0
    colors = np.ones((8, 3), dtype=np.uint16)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
        )


def test_a_colour_on_the_target_grid_is_written():
    """The same pair of conventions carries a value that does convert back exactly, so what save refuses is the loss rather than the narrowing."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    # 257 converts to 1 and back to 257
    colors = np.full((8, 3), 257, dtype=np.uint16)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    save_point_cloud(
        pc,
        filepath,
        meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    for column_name in ('red', 'green', 'blue'):
        np.testing.assert_array_equal(
            vertex_data[column_name], np.ones(8, dtype=np.uint8)
        )


def test_an_ordinary_field_narrowing_out_of_range_is_refused_too():
    """A colour reaches its target through a range mapping and an ordinary field through a dtype cast, and both refuse the value they cannot carry back."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'intensity': np.full(8, 300, dtype=np.uint16),
        },
        device='cpu',
    )

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={'intensity': {'dtype': 'uint8', 'layout': ('intensity',)}},
        )


def test_a_colour_target_ply_cannot_carry_is_refused():
    """A colour target's range is the convention the conversion fills, so save refuses instead of scaling onto a range the column cannot hold."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    colors = np.tile(np.array([10, 128, 250], dtype=np.uint8), (8, 1))
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'rgb': colors,
        },
        device='cpu',
    )

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={'rgb': {'dtype': 'int64', 'layout': ('red', 'green', 'blue')}},
        )


def test_a_float_rgb_outside_zero_to_one_is_refused():
    """What save asserts is that the values sit inside the range the field's current dtype declares."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    with pytest.raises(AssertionError):
        coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
        pc = PointCloud(
            data={
                'x': coordinates[:, 0],
                'y': coordinates[:, 1],
                'z': coordinates[:, 2],
                'rgb': np.full((8, 3), 1.5, dtype=np.float64),
            },
            device='cpu',
        )
        save_point_cloud(
            pc,
            filepath,
            meta_data={'rgb': {'dtype': 'uint8', 'layout': ('red', 'green', 'blue')}},
        )


def test_a_field_is_written_under_the_column_names_its_meta_data_holds(pc):
    """The output column names come from the meta data's layout, never from the field name, so a one-column field lands under the name its source gave it."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    values = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    pc.feat = values

    save_point_cloud(
        pc,
        filepath,
        meta_data={'feat': {'dtype': 'float32', 'layout': ('intensity',)}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    np.testing.assert_array_equal(vertex_data['intensity'], values.squeeze(-1).numpy())
    assert (
        'feat' not in vertex_data.dtype.names
    ), f"a field is written under the names its layout holds and never under its own: columns={vertex_data.dtype.names}"


def test_a_multi_column_field_takes_one_meta_data_name_per_column(pc):
    """A field of several columns is written under the several names its layout gives, one column per name."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    pc.feat = values

    save_point_cloud(
        pc,
        filepath,
        meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b', 'c')}},
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    for index, column_name in enumerate(('a', 'b', 'c')):
        np.testing.assert_array_equal(
            vertex_data[column_name], values[:, index].numpy()
        )


def test_a_layout_naming_the_wrong_number_of_columns_is_refused():
    """A mapping that does not name exactly as many source columns as the field carries leaves save with no names to write under."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
        },
        device='cpu',
    )
    pc.feat = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={'feat': {'dtype': 'float32', 'layout': ('a', 'b')}},
        )


def test_a_multi_column_identity_layout_is_refused_by_the_column_count():
    """An in-memory three-column field gets one name standing for the whole block, which is one name against three ply columns."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    pc = PointCloud(
        xyz=torch.arange(24, dtype=torch.float32).reshape(8, 3), device='cpu'
    )

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath)


def test_a_field_the_meta_data_does_not_name_takes_its_target_from_itself():
    """A field assigned after construction is outside the meta data and writes under its own name in the width it means."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        source_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    rows['x'] = np.arange(8, dtype=np.float32)
    PlyData([PlyElement.describe(rows, 'vertex')]).write(source_path)

    pc = load_point_cloud(filepath=source_path, device='cpu')
    pc.visible = torch.arange(8, dtype=torch.uint8).unsqueeze(-1) % 2

    save_point_cloud(pc, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['visible'] == np.dtype(
        'u1'
    ), f"a uint8 mask reaches a u1 column: stored dtype={vertex_data.dtype['visible']}"


def test_a_field_the_meta_data_names_that_the_cloud_dropped_is_not_written():
    """Deleting a field leaves the meta data naming it, and save walks the cloud's own fields, so the departed field reaches no column."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        source_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    rows = np.zeros(
        8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
    )
    rows['intensity'] = np.arange(8, dtype=np.float32)
    PlyData([PlyElement.describe(rows, 'vertex')]).write(source_path)

    pc = load_point_cloud(filepath=source_path, device='cpu')
    del pc.intensity

    save_point_cloud(pc, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert (
        'intensity' not in vertex_data.dtype.names
    ), f"a field the cloud no longer holds reaches no column: columns={vertex_data.dtype.names}"


def test_a_pcd_loaded_field_needs_a_layout_to_reach_ply_columns():
    """A pcd attribute is one named block, so the caller names the columns a ply save writes under."""
    with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as tmp_file:
        pcd_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    write_pcd(pcd_path, with_colors=True)

    loaded = load_point_cloud(filepath=pcd_path, device='cpu')

    with pytest.raises(AssertionError):
        save_point_cloud(loaded, filepath)

    save_point_cloud(
        loaded,
        filepath,
        meta_data={
            'xyz': {'layout': ('x', 'y', 'z')},
            'rgb': {'layout': ('red', 'green', 'blue')},
        },
    )
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert set(vertex_data.dtype.names) == {
        'x',
        'y',
        'z',
        'red',
        'green',
        'blue',
    }, f"a caller-stated layout names the columns a pcd-loaded cloud reaches: columns={vertex_data.dtype.names}"


def test_a_layout_repeating_a_column_is_refused(pc):
    """A caller-stated layout never passes through a meta data entry, so its own distinctness is checked at the door it comes in by."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    pc.feat = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath, meta_data={'feat': {'layout': ('a', 'a', 'b')}})


def test_two_fields_writing_one_column_are_refused(pc):
    """Two layouts naming the same ply column would silently overwrite each other and then die inside numpy on a duplicate field name."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    pc.feat = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    pc.label = torch.arange(8, dtype=torch.float32).unsqueeze(-1)

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={
                'feat': {'layout': ('shared',)},
                'label': {'layout': ('shared',)},
            },
        )


def test_a_dtype_naming_no_such_field_is_refused_at_save(pc):
    """Save walks the cloud's own fields, so a key naming none of them would leave the meta data silently in force."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath, meta_data={'nosuchfield': {'dtype': 'float32'}})


def test_a_layout_naming_no_such_field_is_refused(pc):
    """A name the cloud does not carry aborts rather than leaving the meta data's own columns silently in force."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath, meta_data={'nosuchfield': {'layout': ('a',)}})


def test_a_value_that_is_not_a_point_cloud_is_refused():
    """What this door refuses is a value that is not a PointCloud at all."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    with pytest.raises(AssertionError):
        save_point_cloud('not a point cloud', filepath)


def test_wrong_file_extension_error(pc):
    """An output path no writer owns is rejected the same way the load door rejects one."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath.replace('.ply', '.xyz'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tensor_saving():
    """Fields living on a cuda device are written from there."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = torch.arange(24, dtype=torch.float32, device='cuda').reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
        },
        device='cuda',
    )

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.xyz.cpu().numpy(), coordinates.cpu().numpy())


def test_mixed_tensor_types_saving():
    """One PointCloud may carry numpy and torch fields together."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    features = torch.arange(8, dtype=torch.float32).unsqueeze(-1)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'feat': features,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.xyz.cpu().numpy(), coordinates)
    np.testing.assert_array_equal(loaded.feat.cpu().numpy(), features.numpy())


def test_a_pth_loaded_field_writes_back_under_the_index_names_its_meta_data_holds():
    """A .pth names its columns by position, so a ply written from one carries columns called 0, 1 and 2 until a caller states otherwise."""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        pth_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    torch.save(torch.arange(24, dtype=torch.float32).reshape(8, 3), pth_path)

    loaded = load_point_cloud(
        filepath=pth_path, meta_data={'xyz': {'layout': ('0', '1', '2')}}, device='cpu'
    )

    save_point_cloud(loaded, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype.names == (
        '0',
        '1',
        '2',
    ), f"a positional source writes back under the names it gave its columns: columns={vertex_data.dtype.names}"

    save_point_cloud(loaded, filepath, meta_data={'xyz': {'layout': ('x', 'y', 'z')}})
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype.names == (
        'x',
        'y',
        'z',
    ), f"naming the columns for another reader is the caller's to ask for: columns={vertex_data.dtype.names}"


@pytest.mark.parametrize(
    "coordinates,features",
    [
        (np.arange(24, dtype=np.float32).reshape(8, 3), None),
        (
            (537000.0 + np.arange(24, dtype=np.float32)).reshape(8, 3),
            None,
        ),
        (
            np.arange(24, dtype=np.float32).reshape(8, 3),
            np.arange(8, dtype=np.float32).reshape(8, 1),
        ),
    ],
)
def test_save_load_round_trip(coordinates, features):
    """Across coordinate magnitudes, and with a feature column or with coordinates alone, saving then loading preserves the values."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    columns = {
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'z': coordinates[:, 2],
    }
    if features is not None:
        columns['feat'] = features
    pc = PointCloud(data=columns, device='cpu')

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.xyz.cpu().numpy(), coordinates)


def test_indices_survive_the_ply_round_trip():
    """A PointCloud carrying indices comes back carrying them, in the width the meta data names."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    indices = np.arange(8, dtype=np.int64)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'indices': indices,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    np.testing.assert_array_equal(loaded.indices.cpu().numpy().reshape(-1), indices)
    assert (
        loaded.indices.dtype == torch.int32
    ), f"ply carries no 64-bit integer, so the width that comes back is the one its i4 column names: indices.dtype={loaded.indices.dtype}"


def test_an_int64_target_goes_to_an_i4_column():
    """ply has no 64-bit integer, so the ply column takes the largest narrower dtype it carries, with the values deciding."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    labels = np.arange(8, dtype=np.int64)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'label': labels,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'int64'}})
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['label'] == np.dtype(
        'i4'
    ), f"an int64 target reaches an i4 column: stored dtype={vertex_data.dtype['label']}"
    np.testing.assert_array_equal(vertex_data['label'], labels)


def test_a_uint64_target_goes_to_a_u4_column():
    """An unsigned target keeps an unsigned column, which is what the signedness tie-break among equal-width narrowing candidates is for."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'label': np.arange(8, dtype=np.int64),
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'uint64'}})
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['label'] == np.dtype(
        'u4'
    ), f"a uint64 target reaches a u4 column: stored dtype={vertex_data.dtype['label']}"


def test_an_int64_target_whose_values_exceed_i4_is_refused():
    """ply carries no int64, so this aborts inside the narrowing: the target is narrowed to i4 and the values are tested against that candidate."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'label': np.full(8, 2**40, dtype=np.int64),
        },
        device='cpu',
    )

    with pytest.raises(AssertionError):
        save_point_cloud(pc, filepath, meta_data={'label': {'dtype': 'int64'}})


def test_a_cast_that_would_lose_a_value_is_refused():
    """ply carries i4 outright, so nothing narrows and this aborts inside the cast instead."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name

    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'label': np.full(8, 2**40, dtype=np.int64),
        },
        device='cpu',
    )

    with pytest.raises(AssertionError):
        save_point_cloud(
            pc,
            filepath,
            meta_data={'label': {'dtype': 'int32', 'layout': ('label',)}},
        )


def test_a_u2_column_round_trips_back_to_u2():
    """A field loaded from a u2 column is stored as int32 and written back as u2, because save follows the meta data rather than the stored dtype."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        source_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u2')])
    rows['label'] = np.arange(8, dtype=np.uint16) * 1000
    PlyData([PlyElement.describe(rows, 'vertex')]).write(source_path)

    loaded = load_point_cloud(filepath=source_path, device='cpu')

    save_point_cloud(loaded, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['label'] == np.dtype(
        'u2'
    ), f"a u2 source column reaches a u2 column again: stored dtype={vertex_data.dtype['label']}"
    np.testing.assert_array_equal(vertex_data['label'], rows['label'])


def test_a_u4_column_round_trips_back_to_u4():
    """A u4 column stored as int64 is written back as u4, which is the whole point of noting the source dtype."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        source_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('id', 'u4')])
    rows['id'] = np.arange(8, dtype=np.uint32) * 100000
    PlyData([PlyElement.describe(rows, 'vertex')]).write(source_path)

    loaded = load_point_cloud(filepath=source_path, device='cpu')

    save_point_cloud(loaded, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['id'] == np.dtype(
        'u4'
    ), f"a u4 source column reaches a u4 column again: stored dtype={vertex_data.dtype['id']}"
    np.testing.assert_array_equal(vertex_data['id'], rows['id'])


def test_xyz_is_written_in_the_width_its_meta_data_names():
    """Coordinates are no longer written as f4 whatever they are: an f8 source goes back out as f8."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        source_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    rows = np.zeros(8, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    rows['x'] = 537000.123456 + np.arange(8, dtype=np.float64)
    PlyData([PlyElement.describe(rows, 'vertex')]).write(source_path)

    loaded = load_point_cloud(filepath=source_path, device='cpu')

    save_point_cloud(loaded, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    for column_name in ('x', 'y', 'z'):
        assert vertex_data.dtype[column_name] == np.dtype(
            'f8'
        ), f"an f8 source column reaches an f8 column again: column_name={column_name}, stored dtype={vertex_data.dtype[column_name]}"


def test_precision_consistency_save_load(pc):
    """The precision the writer keeps is the precision the reader hands back."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = pc.xyz.cpu().numpy().copy()

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    error = np.abs(loaded.xyz.cpu().numpy() - coordinates).max()
    assert (
        error <= torch.finfo(torch.float32).eps * np.abs(coordinates).max()
    ), f"coordinates come back within float32's own resolution of the magnitude: error={error}, magnitude={np.abs(coordinates).max()}, eps={torch.finfo(torch.float32).eps}"


def test_a_bool_field_widens_to_the_unsigned_byte_column():
    """ply declares no boolean type at all, so a mask reaches a one-byte integer column, and u1 is the one of the two whose signedness matches bool's."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    mask = np.arange(8) % 2 == 0
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'visible': mask,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['visible'] == np.dtype(
        'u1'
    ), f"a bool field reaches a u1 column: stored dtype={vertex_data.dtype['visible']}"
    np.testing.assert_array_equal(vertex_data['visible'], mask.astype(np.uint8))


def test_a_bool_field_comes_back_from_ply_as_an_integer():
    """The widening is the one place a ply round trip does not return the dtype it took, and the meta data stays truthful about what the file it just read actually holds."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    mask = np.arange(8) % 2 == 0
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'visible': mask,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    loaded = load_point_cloud(filepath=filepath, device='cpu')

    assert (
        loaded.visible.dtype == torch.uint8
    ), f"a bool field comes back as the integer the column holds: visible.dtype={loaded.visible.dtype}"
    assert (
        loaded.meta_data['visible']['dtype'] == 'uint8'
    ), f"the meta data names what the file it just read holds: entry={loaded.meta_data['visible']}"
    np.testing.assert_array_equal(
        loaded.visible.cpu().numpy().reshape(-1), mask.astype(np.uint8)
    )


def test_a_bfloat16_field_widens_to_leave_torch():
    """numpy holds no bfloat16 at all, so the field widens to the narrowest numpy dtype containing it before it can become a ply column."""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        filepath = tmp_file.name
    coordinates = np.arange(24, dtype=np.float32).reshape(8, 3)
    features = torch.arange(8, dtype=torch.bfloat16).unsqueeze(-1)
    pc = PointCloud(
        data={
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'z': coordinates[:, 2],
            'feature': features,
        },
        device='cpu',
    )

    save_point_cloud(pc, filepath)
    vertex_data = PlyData.read(filepath)['vertex'].data

    assert vertex_data.dtype['feature'] == np.dtype(
        'f4'
    ), f"a bfloat16 field reaches an f4 column: stored dtype={vertex_data.dtype['feature']}"
    np.testing.assert_array_equal(
        vertex_data['feature'], features.float().numpy().reshape(-1)
    )
