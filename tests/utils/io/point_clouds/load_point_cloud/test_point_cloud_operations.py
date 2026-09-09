import os
import tempfile
from typing import Optional

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
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


def test_values_survive_the_load(temp_dir):
    """The coordinates that come back are the ones that were saved."""
    filepath = os.path.join(temp_dir, "known.pth")
    saved = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    torch.save(saved, filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}},
        device='cpu',
    )

    assert torch.equal(result.xyz, saved), f"{result.xyz=}, {saved=}"


def test_placed_on_the_requested_device(temp_dir):
    """Every field lands on the device the caller named."""
    filepath = os.path.join(temp_dir, "device.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cpu',
    )

    for name in result.field_names():
        assert (
            getattr(result, name).device.type == 'cpu'
        ), f"{name=}, {getattr(result, name).device=}"


def test_placed_on_cuda_when_asked(temp_dir):
    """The same placement holds for a cuda device, where one is available."""
    if not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    filepath = os.path.join(temp_dir, "cuda.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cuda',
    )

    assert result.xyz.device.type == 'cuda', f"{result.xyz.device=}"


def test_a_dtype_reaches_one_field_at_a_time(temp_dir):
    """The dtype half reaches one field by name, leaving every other field at the dtype its source held."""
    filepath = os.path.join(temp_dir, "one_field.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'dtype': 'float64', 'layout': ('0', '1', '2')},
            'feat': {'layout': ('3',)},
        },
        device='cpu',
    )

    assert result.xyz.dtype == torch.float64, f"{result.xyz.dtype=}"
    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"
    assert result.meta_data['xyz']['dtype'] == 'float32', f"{result.meta_data['xyz']=}"


def test_a_dtype_reaches_a_field_the_coordinates_are_not(temp_dir):
    """The retired dtype argument could only cast the coordinates; the dtype half reaches any field by name."""
    filepath = os.path.join(temp_dir, "other_field.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={
            'xyz': {'layout': ('0', '1', '2')},
            'feat': {'dtype': 'float64', 'layout': ('3',)},
        },
        device='cpu',
    )

    assert result.feat.dtype == torch.float64, f"{result.feat.dtype=}"
    assert result.xyz.dtype == torch.float32, f"{result.xyz.dtype=}"


def test_a_column_no_layout_names_is_not_loaded(temp_dir):
    """A caller writing the layout by hand has chosen which columns become fields, so a column none of them names is simply absent rather than assembled into a field of its own."""
    filepath = os.path.join(temp_dir, "seven.pth")
    torch.save(torch.rand(size=(8, 7), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}},
        device='cpu',
    )

    assert result.field_names() == ('xyz',), f"{result.field_names()=}"


def test_a_bfloat16_target_is_reached_from_a_float32_source(temp_dir):
    """The dtype a caller states may name the one dtype only torch carries, and the cast runs in the system that has it."""
    filepath = os.path.join(temp_dir, "bfloat16.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'intensity': {'dtype': 'bfloat16'}},
        device='cpu',
    )

    assert result.intensity.dtype == torch.bfloat16, f"{result.intensity.dtype=}"


def test_a_stated_colour_dtype_moves_the_values_onto_the_convention_it_names(temp_dir):
    """A colour's dtype IS its convention, so stating one converts the values onto that range and the record then says the convention they are on."""
    filepath = os.path.join(temp_dir, "convention.ply")
    write_ply(
        filepath,
        with_rgb=True,
        colour_columns='uint16',
        colour_values='multiples of 257',
    )

    result = load_point_cloud(
        filepath=filepath, meta_data={'rgb': {'dtype': 'uint8'}}, device='cpu'
    )

    assert result.rgb.dtype == torch.uint8, f"{result.rgb.dtype=}"
    assert torch.equal(
        result.rgb,
        torch.tensor([[0, 128, 255]] * result.rgb.shape[0], dtype=torch.uint8),
    ), f"{result.rgb=}"
    assert result.meta_data['rgb']['dtype'] == 'uint8', f"{result.meta_data['rgb']=}"


def test_a_stated_colour_dtype_that_would_round_the_values_is_refused(temp_dir):
    """Construction and load refuse a lossy conversion exactly as save does, so a colour that cannot come back is never quietly rounded on the way in."""
    filepath = os.path.join(temp_dir, "lossy_convention.ply")
    write_ply(filepath, with_rgb=True, colour_columns='uint16', colour_values='1')

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath, meta_data={'rgb': {'dtype': 'uint8'}}, device='cpu'
        )


def test_a_stated_dtype_that_would_narrow_a_value_away_is_refused(temp_dir):
    """Every cast these modules make is lossless, so narrowing is the caller's to do on its own values before handing them in."""
    filepath = os.path.join(temp_dir, "narrowing.pth")
    coordinates = torch.tensor(
        [
            [500000.123456789, 4000000.987654321, 123.456789012, 1.0],
            [500001.123456789, 4000001.987654321, 124.456789012, 2.0],
        ],
        dtype=torch.float64,
    )
    torch.save(coordinates, filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'dtype': 'float32', 'layout': ('0', '1', '2')}},
            device='cpu',
        )


def test_a_dtype_alone_is_enough_where_the_source_defines_the_layout(temp_dir):
    """A ply names its own columns, so a caller who only wants a different dtype states only that and the file's layout stands."""
    filepath = os.path.join(temp_dir, "dtype_only.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'intensity': {'dtype': 'float64'}},
        device='cpu',
    )

    assert result.intensity.dtype == torch.float64, f"{result.intensity.dtype=}"
    assert result.meta_data['xyz']['layout'] == (
        'x',
        'y',
        'z',
    ), f"{result.meta_data['xyz']=}"


def test_a_meta_entry_naming_no_such_field_is_refused_at_load(temp_dir):
    """A name the layout never produces aborts rather than leaving the source dtype silently in force, and the layout it does produce is stated so the misspelling is what fails."""
    filepath = os.path.join(temp_dir, "misspelled.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={
                'xyz': {'layout': ('0', '1', '2')},
                'feat': {'layout': ('3',)},
                'nosuchfield': {'dtype': 'float64'},
            },
            device='cpu',
        )


def test_a_seg_filename_no_longer_casts_the_feature_column(temp_dir):
    """No filename decides a dtype any more: a basename carrying _seg keeps its feature column exactly as stored."""
    filepath = os.path.join(temp_dir, "cloud_seg.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cpu',
    )

    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_feature_column_is_left_alone_without_the_seg_marker(temp_dir):
    """The same file without _seg in its basename keeps its feature column's dtype."""
    filepath = os.path.join(temp_dir, "cloud.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cpu',
    )

    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_a_u2_column_is_stored_as_int32_and_notes_uint16(temp_dir):
    """torch carries no uint16, so the field is stored in the narrowest torch dtype that holds it while the meta data keeps what the column held."""
    filepath = os.path.join(temp_dir, "u2.ply")
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u2')])
    rows['label'] = np.arange(8, dtype=np.uint16)
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.label.dtype == torch.int32, f"{result.label.dtype=}"
    assert (
        result.meta_data['label']['dtype'] == 'uint16'
    ), f"{result.meta_data['label']=}"


def test_a_u4_column_is_stored_as_int64_and_notes_uint32(temp_dir):
    """The same patch one width up: torch carries no uint32 either, and the meta data still keeps the column's own dtype."""
    filepath = os.path.join(temp_dir, "u4.ply")
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('id', 'u4')])
    rows['id'] = np.arange(8, dtype=np.uint32)
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.id.dtype == torch.int64, f"{result.id.dtype=}"
    assert result.meta_data['id']['dtype'] == 'uint32', f"{result.meta_data['id']=}"


def test_a_field_whose_source_columns_disagree_in_dtype_is_rejected(temp_dir):
    """A field is assembled only from columns that all hold one dtype; a file whose columns disagree aborts rather than being promoted to a dtype covering them all."""
    filepath = os.path.join(temp_dir, "disagreeing.ply")
    rows = np.zeros(8, dtype=[('x', 'f4'), ('y', 'f8'), ('z', 'f8')])
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_no_filename_marker_decides_a_dtype(temp_dir):
    """An uppercase _SEG basename never meant anything, and now neither does the lowercase one."""
    filepath = os.path.join(temp_dir, "cloud_SEG.pth")
    torch.save(torch.rand(size=(8, 4), dtype=torch.float32), filepath)

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}, 'feat': {'layout': ('3',)}},
        device='cpu',
    )

    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_integer_xyz_is_rejected(temp_dir):
    """An integer coordinate block is rejected rather than cast into a valid-looking float one."""
    filepath = os.path.join(temp_dir, "integer.pth")
    torch.save(torch.arange(24, dtype=torch.int64).reshape(8, 3), filepath)

    # the layout is stated so the integer coordinates are what fails
    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '2')}},
            device='cpu',
        )


def test_windows_style_path_resolves(temp_dir):
    """A path written with backslashes names the same file as the one written with slashes."""
    filepath = os.path.join(temp_dir, "windows.ply")
    write_ply(filepath)
    windows_style_path = filepath.replace('/', '\\')

    result = load_point_cloud(filepath=windows_style_path, device='cpu')

    assert isinstance(result, PointCloud), f"{type(result)=}"


def test_sizes_from_one_point_upward(temp_dir):
    """Point clouds of any row count load with their row count preserved."""
    for size in (1, 10, 1000):
        filepath = os.path.join(temp_dir, f"size_{size}.ply")
        write_ply(filepath, num_points=size)

        result = load_point_cloud(filepath=filepath, device='cpu')

        assert result.xyz.shape == (size, 3), f"{size=}, {result.xyz.shape=}"


def test_empty_point_cloud_is_rejected(temp_dir):
    """A file holding no points is rejected rather than yielding an empty PointCloud."""
    filepath = os.path.join(temp_dir, "empty.pth")
    torch.save(torch.zeros(size=(0, 3), dtype=torch.float32), filepath)

    # the layout is stated so the empty file is what fails
    with pytest.raises(AssertionError):
        load_point_cloud(
            filepath=filepath,
            meta_data={'xyz': {'layout': ('0', '1', '2')}},
            device='cpu',
        )


def write_ply(
    filepath: str,
    num_points: int = 8,
    extra_field: Optional[str] = None,
    with_rgb: bool = False,
    colour_columns: str = 'uint8',
    colour_values: str = 'multiples of 257',
) -> np.ndarray:
    """Writes a single-element PLY, since this suite needs a source that defines its own layout beside the .pth ones that define none.

    Args:
        filepath: Destination path of the PLY file, as a str.
        num_points: Number of vertices the file carries, as an int.
        extra_field: Name of one additional float32 column, or None for no extra column.
        with_rgb: Whether the file carries the three colour columns 'red', 'green' and 'blue'.
        colour_columns: The conceptual dtype the colour columns are stored in, as 'uint8' or 'uint16'.
        colour_values: The values every colour row carries, as 'multiples of 257' for the whole 0-to-255 steps 0, 128 and 255 scaled onto the column's own range, or '1' for a value of one in every channel.

    Returns:
        The coordinates the file was written with, as a [num_points, 3] float64 numpy array.
    """
    assert colour_columns in (
        'uint8',
        'uint16',
    ), f"a colour column is written at one of the two widths this suite covers: colour_columns={colour_columns}"
    assert colour_values in (
        'multiples of 257',
        '1',
    ), f"a colour row carries one of the two value sets this suite covers: colour_values={colour_values}"

    coordinates = np.arange(num_points * 3, dtype=np.float64).reshape(num_points, 3)

    fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    values = {name: coordinates[:, axis] for axis, name in enumerate(('x', 'y', 'z'))}
    if with_rgb:
        scale = 1 if colour_columns == 'uint8' else 257
        row = (
            [0 * scale, 128 * scale, 255 * scale]
            if colour_values == 'multiples of 257'
            else [1, 1, 1]
        )
        for index, name in enumerate(('red', 'green', 'blue')):
            fields.append((name, 'u1' if colour_columns == 'uint8' else 'u2'))
            values[name] = np.full(num_points, row[index], dtype=colour_columns)
    if extra_field is not None:
        fields.append((extra_field, 'f4'))
        values[extra_field] = np.arange(num_points, dtype=np.float32)

    rows = np.zeros(num_points, dtype=fields)
    for name, column in values.items():
        rows[name] = column
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)

    return coordinates
