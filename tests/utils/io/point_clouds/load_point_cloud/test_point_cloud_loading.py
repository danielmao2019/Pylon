import os
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import laspy
import numpy as np
import open3d as o3d
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import (
    _load_from_off,
    _load_from_ply,
    _load_from_pth,
    _load_from_txt,
    load_point_cloud,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


@pytest.fixture
def temp_dir():
    """Yields a fresh temporary directory so each test writes its own files.

    Args:
        None.

    Returns:
        Path of a temporary directory removed when the test finishes.
    """
    with tempfile.TemporaryDirectory() as directory:
        yield directory


def test_ply_xyz_only(temp_dir):
    """A PLY carrying only coordinates loads to a PointCloud whose xyz is [N, 3]."""
    filepath = os.path.join(temp_dir, "xyz.ply")
    coordinates = write_ply(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert isinstance(result, PointCloud), f"{type(result)=}"
    assert result.xyz.shape == (
        coordinates.shape[0],
        3,
    ), f"{result.xyz.shape=}, {coordinates.shape=}"
    assert result.xyz.dtype == torch.float32, f"{result.xyz.dtype=}"


def test_ply_with_rgb(temp_dir):
    """RGB columns arrive as an rgb field in the dtype the file stored them in."""
    filepath = os.path.join(temp_dir, "rgb.ply")
    coordinates = write_ply(filepath, with_rgb=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.rgb.shape == (
        coordinates.shape[0],
        3,
    ), f"{result.rgb.shape=}, {coordinates.shape=}"
    assert result.rgb.dtype == torch.uint8, f"{result.rgb.dtype=}"


def test_ply_extra_field_keeps_its_own_name(temp_dir):
    """A non-standard PLY column is loaded under the name the file gives it, not renamed to feat."""
    filepath = os.path.join(temp_dir, "intensity.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'intensity' in result.field_names(), f"{result.field_names()=}"
    assert 'feat' not in result.field_names(), f"{result.field_names()=}"


def test_a_partial_colour_set_stays_separate_columns(temp_dir):
    """The colour mapping is all three columns or none, so a file carrying red and green alone yields two fields under their own names rather than an rgb the third column is missing from."""
    filepath = os.path.join(temp_dir, "partial_colour.ply")
    write_ply(filepath, colour_columns=('red', 'green'))

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'red' in result.field_names(), f"{result.field_names()=}"
    assert 'green' in result.field_names(), f"{result.field_names()=}"
    assert 'rgb' not in result.field_names(), f"{result.field_names()=}"
    assert result.meta_data['red']['layout'] == ('red',), f"{result.meta_data['red']=}"


def test_a_layout_assembles_the_columns_it_names(temp_dir):
    """The layout half chooses which source columns a field is assembled from, which is the control that replaced renaming a column to feat."""
    filepath = os.path.join(temp_dir, "layout.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'feat': {'layout': ('intensity',)}},
        device='cpu',
    )

    assert 'feat' in result.field_names(), f"{result.field_names()=}"
    assert result.meta_data['feat']['layout'] == (
        'intensity',
    ), f"{result.meta_data['feat']=}"
    # a caller-stated layout CONSUMES its columns, so the column does not also survive under its own name
    assert 'intensity' not in result.field_names(), f"{result.field_names()=}"


def test_a_field_s_meta_data_names_the_source_columns_it_was_assembled_from(temp_dir):
    """A PLY maps its three coordinate columns onto xyz and its three color columns onto rgb, and the meta data keeps that mapping."""
    filepath = os.path.join(temp_dir, "mapping.ply")
    write_ply(filepath, with_rgb=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.meta_data['xyz']['layout'] == (
        'x',
        'y',
        'z',
    ), f"{result.meta_data['xyz']=}"
    assert result.meta_data['rgb']['layout'] == (
        'red',
        'green',
        'blue',
    ), f"{result.meta_data['rgb']=}"


def test_a_source_defining_no_layout_is_assembled_by_the_caller(temp_dir):
    """A .pth names nothing about its block, so its columns come back under their own indices and the caller's layout says which of them each field is assembled from."""
    filepath = os.path.join(temp_dir, "block.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'label': {'layout': ('3',)}},
        device='cpu',
    )

    assert result.meta_data['xyz']['layout'] == (
        '0',
        '1',
        '2',
    ), f"{result.meta_data['xyz']=}"
    assert result.meta_data['label']['layout'] == (
        '3',
    ), f"{result.meta_data['label']=}"
    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"
    assert result.label.shape == (8, 1), f"{result.label.shape=}"


def test_a_caller_may_assemble_the_same_block_differently(temp_dir):
    """The reader divines nothing from the column count, so one file loads two ways and neither is the reader's choice."""
    filepath = os.path.join(temp_dir, "seven.pth")
    torch.save(torch.rand(size=(8, 7), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'layout': ('0', '1', '2')},
            'rgb': {'layout': ('3', '4', '5')},
            'label': {'layout': ('6',)},
        },
        device='cpu',
    )

    assert result.rgb.shape == (8, 3), f"{result.rgb.shape=}"

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'layout': ('0', '1', '2')},
            'feat': {'layout': ('3', '4', '5', '6')},
        },
        device='cpu',
    )

    assert result.feat.shape == (8, 4), f"{result.feat.shape=}"
    assert 'rgb' not in result.field_names(), f"{result.field_names()=}"


def test_a_source_defining_no_layout_refuses_a_load_that_states_none(temp_dir):
    """A source that defines no layout makes the caller's meta data required rather than optional, so a load without it aborts instead of guessing."""
    filepath = os.path.join(temp_dir, "unstated.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_a_layout_naming_a_column_index_the_block_lacks_is_refused(temp_dir):
    """The caller selects among the columns the block actually holds, so an index past its width aborts rather than assembling a field from nothing."""
    filepath = os.path.join(temp_dir, "narrow.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '9')}},
            device='cpu',
        )


def test_a_layout_repeating_a_column_is_refused_at_load(temp_dir):
    """A caller-stated layout never passes through a meta data entry on the way in either, so its distinctness is checked at this door as well as at save's."""
    filepath = os.path.join(temp_dir, "repeat.ply")
    write_ply(filepath, with_rgb=True)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'feat': {'layout': ('red', 'red')}},
            device='cpu',
        )


def test_an_empty_layout_is_refused_at_load(temp_dir):
    """A field assembled from no source columns at all is not a field, so an empty tuple aborts rather than producing one."""
    filepath = os.path.join(temp_dir, "empty_layout.ply")
    write_ply(filepath, extra_field='intensity')

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath, meta_data={'feat': {'layout': ()}}, device='cpu'
        )


def test_a_layout_naming_a_column_the_file_lacks_is_refused(temp_dir):
    """A layout can only choose among columns the file actually holds, so naming one it does not aborts rather than assembling a field from nothing."""
    filepath = os.path.join(temp_dir, "missing_column.ply")
    write_ply(filepath, extra_field='intensity')

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'feat': {'layout': ('nosuchcolumn',)}},
            device='cpu',
        )


def test_a_file_without_coordinate_columns_is_refused(temp_dir):
    """Every format's layout starts from its coordinate columns, so a file that names none of them is not a point cloud this reader can build."""
    filepath = os.path.join(temp_dir, "no_coordinates.ply")
    write_ply(filepath, without_coordinates=True)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_a_ply_carrying_no_element_is_refused(temp_dir):
    """A PLY's columns are its elements' properties, so a file declaring no element at all has none and is refused where it is read."""
    filepath = os.path.join(temp_dir, "no_element.ply")
    write_ply(filepath, without_elements=True)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_a_pth_holding_something_other_than_a_block_is_refused(temp_dir):
    """A .pth is one block of columns, so a payload that is a dict or a list is refused rather than being indexed as though it were an array."""
    filepath = os.path.join(temp_dir, "mapping.pth")
    torch.save({'not': 'a block'}, filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '2')}},
            device='cpu',
        )


def test_a_pth_holding_a_block_of_one_axis_is_refused(temp_dir):
    """Columns are keyed by index along a second axis, so a block that has none is refused at this door rather than raising an IndexError from inside the split."""
    filepath = os.path.join(temp_dir, "flat.pth")
    torch.save(torch.rand(size=(8,), dtype=torch.float32), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '2')}},
            device='cpu',
        )


def test_a_pcd_without_positions_is_refused(temp_dir):
    """Open3D names the coordinate attribute positions, so a pcd carrying none of it is not a point cloud this reader can build."""
    filepath = os.path.join(temp_dir, "no_positions.pcd")
    write_pcd(filepath, without_positions=True)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_a_las_bit_packed_dimension_enters_as_an_ordinary_uint8_field(temp_dir):
    """laspy materializes a bit-packed dimension as uint8, so uint8 is what enters the obj and what the meta data stores, with no special treatment."""
    filepath = os.path.join(temp_dir, "classified.las")
    write_las(filepath, with_classification=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.return_number.dtype == torch.uint8, f"{result.return_number.dtype=}"
    assert (
        result.meta_data['return_number']['dtype'] == 'uint8'
    ), f"{result.meta_data['return_number']=}"
    assert result.meta_data['return_number']['layout'] == (
        'return_number',
    ), f"{result.meta_data['return_number']=}"


def test_a_las_uint16_colour_keeps_its_own_width(temp_dir):
    """las stores colours as uint16, and a reader neither widens nor narrows, so the meta data holds uint16 while torch parks it in int32."""
    filepath = os.path.join(temp_dir, "coloured.las")
    write_las(filepath, with_rgb=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.rgb.dtype == torch.int32, f"{result.rgb.dtype=}"
    # the meta data and the storage differ here, which is the divergence a display reads the wrong side of when it reads the tensor
    assert result.meta_data['rgb']['dtype'] == 'uint16', f"{result.meta_data['rgb']=}"
    assert result.meta_data['rgb']['layout'] == (
        'red',
        'green',
        'blue',
    ), f"{result.meta_data['rgb']=}"


def test_a_las_dimension_the_file_does_not_carry_is_skipped(temp_dir):
    """A point format that has no colour does not name red, green and blue among its dimensions at all, so they become no column rather than an empty one."""
    filepath = os.path.join(temp_dir, "colourless.las")
    write_las(filepath, with_rgb=False)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'rgb' not in result.field_names(), f"{result.field_names()=}"
    assert 'xyz' in result.field_names(), f"{result.field_names()=}"
    assert 'return_number' in result.field_names(), f"{result.field_names()=}"


def test_a_las_coordinate_is_the_scaled_value_not_the_raw_dimension(temp_dir):
    """A las stores its coordinates as integers to be multiplied by the header's scale and shifted by its offset, so the dimension's own values are not points and reading them as points misplaces every one."""
    filepath = os.path.join(temp_dir, "scaled.las")
    coordinates = write_las(
        filepath, scales=(0.001, 0.001, 0.001), offsets=(100.0, 200.0, 300.0)
    )

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert np.allclose(
        result.xyz.numpy(), coordinates, atol=1e-9
    ), f"{result.xyz.numpy()=}, {coordinates=}"
    # the coordinate's own dtype is the scaled float and its name is the one laspy gives it, while the raw int32 dimension is the container's storage of both, the same divergence a uint16 colour has inside an int32 tensor
    assert result.meta_data['xyz']['dtype'] == 'float64', f"{result.meta_data['xyz']=}"
    assert result.meta_data['xyz']['layout'] == (
        'x',
        'y',
        'z',
    ), f"{result.meta_data['xyz']=}"


def test_a_laz_file_loads_as_its_las_counterpart_does(temp_dir):
    """The compressed container changes nothing about the dimensions laspy hands back."""
    filepath = os.path.join(temp_dir, "compressed.laz")
    coordinates = write_las(filepath, with_rgb=True, compressed=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert np.allclose(
        result.xyz.numpy(), coordinates, atol=1e-9
    ), f"{result.xyz.numpy()=}, {coordinates=}"
    assert result.meta_data['rgb']['dtype'] == 'uint16', f"{result.meta_data['rgb']=}"


def test_a_pcd_names_its_own_layout_from_its_attributes(temp_dir):
    """A pcd's attributes are named, so each one is a source column in its own right and the reader states the layout over those names."""
    filepath = os.path.join(temp_dir, "attributes.pcd")
    write_pcd(filepath, with_colors=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"
    assert result.rgb.shape == (8, 3), f"{result.rgb.shape=}"
    assert result.meta_data['xyz']['layout'] == (
        'positions',
    ), f"{result.meta_data['xyz']=}"
    assert result.meta_data['rgb']['layout'] == (
        'colors',
    ), f"{result.meta_data['rgb']=}"


def test_a_pcd_attribute_beyond_positions_and_colors_keeps_its_own_name(temp_dir):
    """Only the two Open3D names for coordinates and colour are renamed; every other attribute names its own field."""
    filepath = os.path.join(temp_dir, "extra_attribute.pcd")
    write_pcd(filepath, with_colors=True, extra_attribute='intensity')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert set(result.field_names()) == {
        'xyz',
        'rgb',
        'intensity',
    }, f"{result.field_names()=}"


def test_a_pcd_colour_arrives_as_uint8_whatever_the_writer_held(temp_dir):
    """PCD stores colour as bytes, so Open3D scales a float colour by 255 on the way out and the field that comes back means the 0-to-255 convention rather than the 0-to-1 one it was written in."""
    filepath = os.path.join(temp_dir, "float_colour.pcd")
    write_pcd(filepath, with_colors=True, colors_dtype='float32')

    result = load_point_cloud(filepath=filepath, device='cpu')

    # the meta data states the file, not the writer's intent, which is what keeps the convention read off it truthful
    assert result.meta_data['rgb']['dtype'] == 'uint8', f"{result.meta_data['rgb']=}"
    assert int(result.rgb.max()) > 1, f"{result.rgb=}"


def test_a_pcd_unsigned_attribute_reaches_the_cloud_at_its_own_width(temp_dir):
    """Open3D carries unsigned widths torch has none of, so the reader's route to numpy is what decides whether a uint16 attribute arrives as uint16 or not at all."""
    filepath = os.path.join(temp_dir, "unsigned_attribute.pcd")
    write_pcd(
        filepath,
        with_colors=True,
        extra_attribute='intensity',
        extra_attribute_dtype='uint16',
    )

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert (
        result.meta_data['intensity']['dtype'] == 'uint16'
    ), f"{result.meta_data['intensity']=}"
    # a route through torch raises before reaching this, since torch names no unsigned width above uint8
    assert result.intensity.flatten().tolist() == list(range(8)), f"{result.intensity=}"


def test_ply_custom_element_name(temp_dir):
    """The one element a PLY carries is the file's to name, so an element called anything at all is the one that is read."""
    filepath = os.path.join(temp_dir, "named_element.ply")
    write_ply(filepath, element_name='points')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"


def test_a_multi_element_ply_is_assembled_from_the_layouts_the_caller_names(temp_dir):
    """A file with more than one element does not define which element's columns a field comes from, so the layout the caller states names them and the columns are addressed element-qualified."""
    filepath = os.path.join(temp_dir, "two_elements.ply")
    write_ply(filepath, extra_element=True)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('vertex.x', 'vertex.y', 'vertex.z')}},
        device='cpu',
    )

    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"
    assert result.meta_data['xyz']['layout'] == (
        'vertex.x',
        'vertex.y',
        'vertex.z',
    ), f"{result.meta_data['xyz']=}"


def test_a_multi_element_ply_refuses_a_load_that_states_no_layout(temp_dir):
    """Without the caller naming them there is nothing to choose between two elements' columns, so the load aborts rather than picking one."""
    filepath = os.path.join(temp_dir, "two_elements_unstated.ply")
    write_ply(filepath, extra_element=True)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_txt_columns_are_assembled_only_by_the_caller(temp_dir):
    """The text reader decides nothing about which columns are coordinates, colours or labels; the layout the caller states does, so the same file loads differently for different callers and no width is divined."""
    filepath = os.path.join(temp_dir, "seven.txt")
    written = write_txt(filepath, num_columns=7)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'layout': ('0', '1', '2')},
            'rgb': {'layout': ('3', '4', '5')},
            'label': {'layout': ('6',)},
        },
        device='cpu',
    )

    assert np.allclose(
        result.rgb.numpy(), written[:, 3:6]
    ), f"{result.rgb.numpy()=}, {written[:, 3:6]=}"
    assert np.allclose(
        result.label.numpy(), written[:, 6:7]
    ), f"{result.label.numpy()=}, {written[:, 6:7]=}"


def test_txt_columns_the_caller_names_as_a_colour_are_validated_as_one(temp_dir):
    """Naming three float columns rgb is the caller asking for a colour field, so the 0-to-1 rule applies to them and a 0-to-255 file aborts on the caller's own naming rather than on a guess the reader made."""
    filepath = os.path.join(temp_dir, "byte_range.txt")
    write_txt(filepath, num_columns=7, color_scale=255)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={
                'xyz': {'layout': ('0', '1', '2')},
                'rgb': {'layout': ('3', '4', '5')},
                'label': {'layout': ('6',)},
            },
            device='cpu',
        )


def test_txt_columns_the_caller_names_as_features_are_not(temp_dir):
    """The same file loads without complaint when the caller names those columns as an ordinary feature field, since no colour convention is claimed for them."""
    filepath = os.path.join(temp_dir, "byte_range_as_feature.txt")
    write_txt(filepath, num_columns=7, color_scale=255)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'layout': ('0', '1', '2')},
            'feat': {'layout': ('3', '4', '5', '6')},
        },
        device='cpu',
    )

    assert result.feat.shape == (8, 4), f"{result.feat.shape=}"


def test_pth_tensor(temp_dir):
    """A saved torch tensor's columns are assembled by the caller, since the file names nothing about them."""
    filepath = os.path.join(temp_dir, "tensor.pth")
    write_pth(filepath, torch.rand(size=(8, 4), dtype=torch.float32))

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cpu',
    )

    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"
    assert result.feat.shape == (8, 1), f"{result.feat.shape=}"


def test_pth_ndarray(temp_dir):
    """A saved numpy array is accepted on the same terms as a tensor."""
    filepath = os.path.join(temp_dir, "ndarray.pth")
    write_pth(filepath, np.random.rand(8, 3).astype(np.float32))

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}},
        device='cpu',
    )

    assert result.xyz.shape == (8, 3), f"{result.xyz.shape=}"


def test_pth_rejects_a_non_array_payload(temp_dir):
    """A .pth holding anything but a tensor or an array is rejected rather than half-read."""
    filepath = os.path.join(temp_dir, "payload.pth")
    write_pth(filepath, {'not': 'an array'})

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '2')}},
            device='cpu',
        )


def test_a_uint64_source_column_is_refused_at_load(temp_dir):
    """uint64 is unsupported as a source dtype whatever its values, so it is refused on the way in rather than narrowed."""
    filepath = os.path.join(temp_dir, "unsigned.pth")
    write_pth(filepath, np.arange(24, dtype=np.uint64).reshape(8, 3))

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'ids': {'layout': ('a', 'b', 'c')}},
            device='cpu',
        )


def test_a_uint64_dtype_is_refused_at_load(temp_dir):
    """And naming it as a target dtype does not make it acceptable either, on the load door as on the constructor's."""
    filepath = os.path.join(temp_dir, "unsigned_target.ply")
    write_ply(filepath, extra_field='intensity')

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'intensity': {'dtype': 'uint64'}},
            device='cpu',
        )


def test_off_xyz_only(temp_dir):
    """An OFF file's vertex block loads as coordinates."""
    filepath = os.path.join(temp_dir, "vertices.off")
    write_off(
        filepath, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (4, 3), f"{result.xyz.shape=}"


def test_off_keeps_only_the_leading_three_columns(temp_dir):
    """A vertex line carrying more than three numbers contributes only its coordinates."""
    filepath = os.path.join(temp_dir, "wide_vertices.off")
    vertices = [
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        [0.0, 1.0, 0.0, 7.0, 8.0, 9.0],
        [0.0, 0.0, 1.0, 10.0, 11.0, 12.0],
    ]
    write_off(filepath, vertices)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (4, 3), f"{result.xyz.shape=}"
    assert np.allclose(
        result.xyz.numpy(), np.array(vertices, dtype=np.float32)[:, :3]
    ), f"{result.xyz.numpy()=}, {vertices=}"


def test_off_coordinates_beyond_float32_are_refused(temp_dir):
    """float32 is the width this format is read at, so a vertex float32 cannot hold exactly aborts rather than the read widening to cover it."""
    filepath = os.path.join(temp_dir, "beyond_float32.off")
    write_off(
        filepath,
        [[1e40, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_off_without_its_header_is_rejected(temp_dir):
    """A file whose first line is not OFF is rejected rather than parsed as vertices."""
    filepath = os.path.join(temp_dir, "headerless.off")
    write_off(
        filepath,
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        header='INVALID',
    )

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_an_off_with_its_counts_glued_to_the_keyword_loads(temp_dir):
    """ModelNet40 writes the keyword and the counts on one line, and refusing that shape would refuse the dataset this design loads .off for."""
    filepath = os.path.join(temp_dir, "glued.off")
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    write_off(filepath, vertices, glue_counts_to_header=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (4, 3), f"{result.xyz.shape=}"
    assert np.allclose(
        result.xyz.numpy(), np.array(vertices, dtype=np.float32)
    ), f"{result.xyz.numpy()=}, {vertices=}"


def test_an_off_with_a_comment_before_its_counts_loads(temp_dir):
    """OFF permits a comment line, so the counts are the next line that carries any, rather than the next line whatever it holds."""
    filepath = os.path.join(temp_dir, "commented.off")
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    write_off(filepath, vertices, comment='made by something')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (4, 3), f"{result.xyz.shape=}"
    assert np.allclose(
        result.xyz.numpy(), np.array(vertices, dtype=np.float32)
    ), f"{result.xyz.numpy()=}, {vertices=}"


def test_load_from_ply_returns_the_file_s_own_columns_as_a_raw_cloud(temp_dir):
    """The PLY reader builds the cloud the file's own column names define, and the record it comes with is how PLY names their layout."""
    filepath = os.path.join(temp_dir, "reader.ply")
    write_ply(filepath, with_rgb=True, extra_field='intensity')

    pc = _load_from_ply(filepath=filepath, device='cpu')

    assert set(pc.field_names()) == {
        'xyz',
        'rgb',
        'intensity',
    }, f"{pc.field_names()=}"
    assert pc.meta_data['xyz']['dtype'] == 'float32', f"{pc.meta_data['xyz']=}"
    assert pc.meta_data['rgb']['dtype'] == 'uint8', f"{pc.meta_data['rgb']=}"
    assert (
        pc.meta_data['intensity']['dtype'] == 'float32'
    ), f"{pc.meta_data['intensity']=}"
    assert pc.meta_data['xyz']['layout'] == (
        'x',
        'y',
        'z',
    ), f"{pc.meta_data['xyz']=}"
    assert pc.meta_data['rgb']['layout'] == (
        'red',
        'green',
        'blue',
    ), f"{pc.meta_data['rgb']=}"
    assert pc.meta_data['intensity']['layout'] == (
        'intensity',
    ), f"{pc.meta_data['intensity']=}"


def test_load_from_txt_returns_its_columns_under_their_indices(temp_dir):
    """The text reader names its columns by position and nothing else, so a seven-column file hands back seven float64 fields and no field names at all."""
    filepath = os.path.join(temp_dir, "reader.txt")
    write_txt(filepath, num_columns=7)

    pc = _load_from_txt(filepath=filepath, device='cpu')

    assert set(pc.field_names()) == {
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
    }, f"{pc.field_names()=}"
    assert all(
        getattr(pc, name).dtype == torch.float64 for name in pc.field_names()
    ), f"{ {name: getattr(pc, name).dtype for name in pc.field_names()} =}"
    # the file defines no coordinate columns, which is what makes the caller's meta data required rather than optional
    assert 'xyz' not in pc.field_names(), f"{pc.field_names()=}"


def test_load_from_txt_divines_no_fields_from_the_column_count(temp_dir):
    """Seven columns named xyz, rgb and feat was the reader's own invention, and the width now decides nothing, which is what the dataset's stated meta data replaced."""
    filepath = os.path.join(temp_dir, "no_divination.txt")
    write_txt(filepath, num_columns=7)

    pc = _load_from_txt(filepath=filepath, device='cpu')

    assert 'xyz' not in pc.field_names(), f"{pc.field_names()=}"
    assert 'rgb' not in pc.field_names(), f"{pc.field_names()=}"
    assert 'feat' not in pc.field_names(), f"{pc.field_names()=}"


def test_load_from_txt_reads_columns_however_they_are_spaced(temp_dir):
    """Point cloud text is written aligned as often as it is written with single spaces, so the reader splits on runs of whitespace rather than on one named space character."""
    filepath = os.path.join(temp_dir, "aligned.txt")
    written = write_txt(filepath, num_columns=3, spacing='aligned')

    aligned_pc = _load_from_txt(filepath=filepath, device='cpu')

    assert set(aligned_pc.field_names()) == {
        '0',
        '1',
        '2',
    }, f"{aligned_pc.field_names()=}"
    assert all(
        np.allclose(
            getattr(aligned_pc, str(index)).squeeze(-1).numpy(), written[:, index]
        )
        for index in range(3)
    ), f"{ {index: getattr(aligned_pc, str(index)) for index in range(3)} =}, {written=}"

    filepath = os.path.join(temp_dir, "single.txt")
    write_txt(filepath, num_columns=3, spacing='single')

    single_pc = _load_from_txt(filepath=filepath, device='cpu')

    # one delimiter reads both files, where naming a single space reads only the second
    assert all(
        torch.equal(getattr(single_pc, str(index)), getattr(aligned_pc, str(index)))
        for index in range(3)
    ), f"{ {index: getattr(single_pc, str(index)) for index in range(3)} =}, { {index: getattr(aligned_pc, str(index)) for index in range(3)} =}"


def test_a_txt_of_one_row_still_has_columns_to_key(temp_dir):
    """numpy drops the column axis for a file holding one row, so a single-point cloud would arrive with scalars where the columns keyed by index should be."""
    filepath = os.path.join(temp_dir, "one_row.txt")
    write_txt(filepath, num_points=1, num_columns=3)

    pc = _load_from_txt(filepath=filepath, device='cpu')

    assert set(pc.field_names()) == {'0', '1', '2'}, f"{pc.field_names()=}"
    assert all(
        len(getattr(pc, name)) == 1 for name in pc.field_names()
    ), f"{ {name: len(getattr(pc, name)) for name in pc.field_names()} =}"


def test_load_from_pth_returns_a_raw_cloud(temp_dir):
    """The .pth reader builds a cloud whose fields are the block's columns under their own indices, its coordinates unnamed until the caller's meta data names them."""
    filepath = os.path.join(temp_dir, "reader.pth")
    write_pth(filepath, torch.rand(size=(8, 4), dtype=torch.float32))

    pc = _load_from_pth(filepath=filepath, device='cpu')

    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert set(pc.field_names()) == {'0', '1', '2', '3'}, f"{pc.field_names()=}"
    assert 'xyz' not in pc.field_names(), f"{pc.field_names()=}"


def test_load_from_off_returns_a_raw_cloud(temp_dir):
    """The OFF format declares its vertex block to be the coordinates, so the reader names them and the cloud it builds already carries xyz."""
    filepath = os.path.join(temp_dir, "reader.off")
    write_off(
        filepath, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    pc = _load_from_off(filepath=filepath, device='cpu')

    assert pc.xyz.shape == (4, 3), f"{pc.xyz.shape=}"
    assert pc.xyz.dtype == torch.float32, f"{pc.xyz.dtype=}"
    assert pc.meta_data['xyz']['layout'] == (
        'x',
        'y',
        'z',
    ), f"{pc.meta_data['xyz']=}"


def test_missing_file_is_rejected():
    """A path naming no file is rejected before any reader is chosen."""
    with pytest.raises(AssertionError):
        load_point_cloud(filepath='nonexistent.ply', device='cpu')


def test_unsupported_extension_is_rejected(temp_dir):
    """An extension no reader owns is rejected, and the file's contents are never opened."""
    filepath = os.path.join(temp_dir, "cloud.xyz")
    with open(filepath, 'w') as f:
        f.write("0.0 0.0 0.0\n")

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_uppercase_extension_is_rejected(temp_dir):
    """An extension differing from a supported one only in case names no reader, so it is rejected."""
    filepath = os.path.join(temp_dir, "upper.PLY")
    write_ply(filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def write_ply(
    filepath: str,
    num_points: int = 8,
    with_rgb: bool = False,
    extra_field: Optional[str] = None,
    element_name: str = 'vertex',
    extra_element: bool = False,
    without_coordinates: bool = False,
    without_elements: bool = False,
    colour_columns: Optional[Tuple[str, ...]] = None,
) -> np.ndarray:
    """Writes a PLY carrying xyz plus whichever optional columns the caller asks for, a second element when one is asked for, and no coordinate columns at all when the caller wants that case.

    Args:
        filepath: Destination path of the PLY file, as a str.
        num_points: Number of rows every element of the file carries, as an int.
        with_rgb: Whether the file carries the three uint8 colour columns 'red', 'green' and 'blue'.
        extra_field: Name of one additional float32 column, or None for no extra column.
        element_name: Name of the element the coordinate columns are written under, as a str.
        extra_element: Whether a second element of the same row count is written beside the first.
        without_coordinates: Whether the coordinate columns 'x', 'y' and 'z' are left out of the file.
        without_elements: Whether the file declares no element at all.
        colour_columns: The colour columns to write as uint8, as a tuple of str naming a subset of 'red', 'green' and 'blue', or None for none.

    Returns:
        The coordinates the file was written with, as a [num_points, 3] float64 numpy array.
    """
    coordinates = np.arange(num_points * 3, dtype=np.float64).reshape(num_points, 3)
    colours = np.arange(num_points * 3, dtype=np.uint8).reshape(num_points, 3)

    fields = []
    values = {}
    if not without_coordinates:
        for axis, name in enumerate(('x', 'y', 'z')):
            fields.append((name, 'f4'))
            values[name] = coordinates[:, axis]
    colour_names = ('red', 'green', 'blue') if with_rgb else (colour_columns or ())
    for index, name in enumerate(colour_names):
        fields.append((name, 'u1'))
        values[name] = colours[:, index]
    if extra_field is not None:
        fields.append((extra_field, 'f4'))
        values[extra_field] = np.arange(num_points, dtype=np.float32)

    rows = np.zeros(num_points, dtype=fields)
    for name, column in values.items():
        rows[name] = column

    elements = [] if without_elements else [PlyElement.describe(rows, element_name)]
    if extra_element:
        extra_rows = np.zeros(num_points, dtype=[('u', 'f4')])
        extra_rows['u'] = np.arange(num_points, dtype=np.float32)
        elements.append(PlyElement.describe(extra_rows, 'other'))
    PlyData(elements).write(filepath)

    return coordinates


def write_txt(
    filepath: str,
    num_points: int = 8,
    num_columns: int = 3,
    spacing: str = 'single',
    color_scale: float = 1,
) -> np.ndarray:
    """Writes a point cloud behind the two header lines the reader skips, spaced the way the caller asks so one delimiter can be shown to read both shapes.

    Args:
        filepath: Destination path of the text file, as a str.
        num_points: Number of rows the table carries, as an int.
        num_columns: Number of columns every row carries, as an int.
        spacing: How the numbers of a row are separated, as 'single' for one space each or 'aligned' for a fixed column width.
        color_scale: The upper bound the values of columns three, four and five span, as a float.

    Returns:
        The values the file was written with, as a [num_points, num_columns] float64 numpy array.
    """
    values = np.arange(num_points * num_columns, dtype=np.float64).reshape(
        num_points, num_columns
    ) / (num_points * num_columns)
    # a caller naming those three as rgb decides which convention they are read under, and the reader itself names none
    for index in (3, 4, 5):
        if index < num_columns:
            values[:, index] = np.linspace(0.0, color_scale, num_points)

    with open(filepath, 'w') as f:
        f.write("header line one\n")
        f.write("header line two\n")
        for row in values:
            if spacing == 'aligned':
                f.write("".join(f"{value:>28.17g}" for value in row) + "\n")
            else:
                f.write(" ".join(f"{value:.17g}" for value in row) + "\n")

    return values


def write_pth(
    filepath: str, array: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
) -> None:
    """Saves one array as the single block a .pth point cloud holds.

    Args:
        filepath: Destination path of the .pth file, as a str.
        array: The payload to save, as a numpy array or a torch tensor of shape [N, C], or any other object when the case under test is a payload that is neither.

    Returns:
        None.
    """
    torch.save(array, filepath)


def write_off(
    filepath: str,
    vertices: Sequence[Sequence[float]],
    header: str = 'OFF',
    glue_counts_to_header: bool = False,
    comment: Optional[str] = None,
) -> None:
    """Writes an OFF file in any of the header shapes the format is found in, the keyword being a parameter so a malformed one can be written too.

    Args:
        filepath: Destination path of the .off file, as a str.
        vertices: The vertex lines to write, as a sequence of sequences of float carrying three numbers or more each.
        header: The keyword the file opens on, as a str.
        glue_counts_to_header: Whether the vertex, face and edge counts are written on the keyword's own line.
        comment: A comment line written between the keyword and the counts, or None for no comment line.

    Returns:
        None.
    """
    with open(filepath, 'w') as f:
        if glue_counts_to_header:
            f.write(f"{header} {len(vertices)} 0 0\n")
        else:
            f.write(f"{header}\n")
            if comment is not None:
                f.write(f"# {comment}\n")
            f.write(f"{len(vertices)} 0 0\n")
        for vertex in vertices:
            f.write(" ".join(repr(float(value)) for value in vertex) + "\n")


def write_las(
    filepath: str,
    num_points: int = 8,
    with_rgb: bool = False,
    with_classification: bool = False,
    compressed: bool = False,
    scales: Tuple[float, float, float] = (0.01, 0.01, 0.01),
    offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Writes a LAS or LAZ file carrying xyz plus, on request, uint16 colour dimensions and the bit-packed return_number laspy materializes as uint8.

    Args:
        filepath: Destination path of the .las or .laz file, as a str.
        num_points: Number of points the file carries, as an int.
        with_rgb: Whether the point format carries the uint16 colour dimensions.
        with_classification: Whether the bit-packed return_number dimension is written.
        compressed: Whether the file is written as LAZ rather than LAS.
        scales: The header's scale for each axis, as a tuple of three floats.
        offsets: The header's offset for each axis, as a tuple of three floats.

    Returns:
        The real-world coordinates the file was written with, as a [num_points, 3] float64 numpy array.
    """
    # a caller pinning these is how the raw dimensions are made to differ visibly from the coordinates they encode
    header = laspy.LasHeader(point_format=2 if with_rgb else 0, version="1.2")
    header.scales = np.array(scales, dtype=np.float64)
    header.offsets = np.array(offsets, dtype=np.float64)
    las_data = laspy.LasData(header)

    coordinates = np.stack(
        [
            offsets[axis] + np.arange(num_points, dtype=np.float64) * scales[axis] * 10
            for axis in range(3)
        ],
        axis=1,
    )
    # the real-world coordinates laspy converts down into the raw ones
    las_data.x = coordinates[:, 0]
    las_data.y = coordinates[:, 1]
    las_data.z = coordinates[:, 2]
    if with_rgb:
        las_data.red = np.full(num_points, 32896, dtype=np.uint16)
        las_data.green = np.full(num_points, 257, dtype=np.uint16)
        las_data.blue = np.full(num_points, 65535, dtype=np.uint16)
    if with_classification:
        las_data.return_number = np.ones(num_points, dtype=np.uint8)
    las_data.write(filepath, do_compress=compressed)

    return coordinates


def write_pcd(
    filepath: str,
    num_points: int = 8,
    with_colors: bool = False,
    colors_dtype: str = 'float32',
    extra_attribute: Optional[str] = None,
    extra_attribute_dtype: str = 'float32',
    without_positions: bool = False,
) -> None:
    """Writes a PCD through Open3D's tensor IO, so its attributes come back as the whole named blocks the reader hands over under their own names.

    Args:
        filepath: Destination path of the .pcd file, as a str.
        num_points: Number of points the file carries, as an int.
        with_colors: Whether the colors attribute is written.
        colors_dtype: The numpy dtype name the colors attribute is written in, as a str.
        extra_attribute: Name of one additional single-column attribute, or None for no extra attribute.
        extra_attribute_dtype: The numpy dtype name the extra attribute is written in, as a str.
        without_positions: Whether the file is written carrying no positions attribute, which Open3D's tensor IO cannot express and which is therefore written as PCD text directly.

    Returns:
        None.
    """
    if without_positions:
        with open(filepath, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\nFIELDS intensity\nSIZE 4\nTYPE F\nCOUNT 1\n")
            f.write(f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\nDATA ascii\n")
            for index in range(num_points):
                f.write(f"{float(index)}\n")
        return

    positions = np.arange(num_points * 3, dtype=np.float32).reshape(num_points, 3)
    point_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(positions))
    if with_colors:
        if colors_dtype == 'float32':
            colors = np.linspace(0.0, 1.0, num_points * 3, dtype=np.float32).reshape(
                num_points, 3
            )
        else:
            colors = (
                np.linspace(0, 255, num_points * 3)
                .astype(colors_dtype)
                .reshape(num_points, 3)
            )
        point_cloud.point['colors'] = o3d.core.Tensor(colors)
    if extra_attribute is not None:
        point_cloud.point[extra_attribute] = o3d.core.Tensor(
            np.arange(num_points, dtype=extra_attribute_dtype).reshape(num_points, 1)
        )
    o3d.t.io.write_point_cloud(filepath, point_cloud)
