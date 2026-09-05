import os
import tempfile
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
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
    """Yield a fresh temporary directory so each test writes its own files.

    Args:
        None.

    Returns:
        Path of a temporary directory removed when the test finishes.
    """
    with tempfile.TemporaryDirectory() as directory:
        yield directory


def write_ply(
    filepath: str,
    num_points: int = 8,
    with_rgb: bool = False,
    extra_field: Optional[str] = None,
    element_name: str = 'vertex',
) -> np.ndarray:
    """Write a PLY whose one element carries xyz plus whichever optional columns the caller asks for.

    Args:
        filepath: Destination path of the PLY file.
        num_points: Number of vertices to write.
        with_rgb: Whether to add uint8 red, green and blue columns.
        extra_field: Name of an additional float32 column, or None for no extra column.
        element_name: Name of the single PLY element.

    Returns:
        The written coordinates as a [num_points, 3] float64 numpy array.
    """
    coordinates = np.stack(
        [
            np.arange(num_points, dtype=np.float64) * 0.1,
            np.arange(num_points, dtype=np.float64) * 0.2,
            np.arange(num_points, dtype=np.float64) * 0.3,
        ],
        axis=1,
    )

    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if with_rgb:
        dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    if extra_field is not None:
        dtype_list.append((extra_field, 'f4'))

    rows: List[Tuple[Any, ...]] = []
    for index in range(num_points):
        row: List[Any] = list(coordinates[index])
        if with_rgb:
            row.extend([index % 255, (index * 2) % 255, (index * 3) % 255])
        if extra_field is not None:
            row.append(float(index))
        rows.append(tuple(row))

    element = PlyElement.describe(np.array(rows, dtype=dtype_list), element_name)
    PlyData([element], text=True).write(filepath)
    return coordinates


def write_txt(filepath: str, num_points: int = 8, num_columns: int = 3) -> np.ndarray:
    """Write a whitespace-separated point cloud behind the two header lines the reader skips.

    Args:
        filepath: Destination path of the text file.
        num_points: Number of rows to write.
        num_columns: Total number of columns per row, the leading three being xyz.

    Returns:
        The written table as a [num_points, num_columns] float64 numpy array.
    """
    table = np.arange(num_points * num_columns, dtype=np.float64).reshape(
        num_points, num_columns
    )
    with open(filepath, 'w') as handle:
        handle.write("# header line 1\n")
        handle.write("# header line 2\n")
        for row in table:
            handle.write(" ".join(str(value) for value in row) + "\n")
    return table


def write_pth(filepath: str, array: Any) -> Any:
    """Save one array as the single block a .pth point cloud holds.

    Args:
        filepath: Destination path of the .pth file.
        array: Payload to save, normally a torch.Tensor or a numpy.ndarray.

    Returns:
        The saved payload, unchanged.
    """
    torch.save(array, filepath)
    return array


def write_off(
    filepath: str, vertices: Sequence[Sequence[float]], header: str = 'OFF'
) -> None:
    """Write an OFF file, the header being a parameter so a malformed one can be written too.

    Args:
        filepath: Destination path of the OFF file.
        vertices: One sequence of floats per vertex, the leading three being xyz.
        header: First line of the file, 'OFF' for a well-formed one.

    Returns:
        None.
    """
    with open(filepath, 'w') as handle:
        handle.write(f"{header}\n")
        handle.write(f"{len(vertices)} 0 0\n")
        for vertex in vertices:
            handle.write(" ".join(str(value) for value in vertex) + "\n")


def test_ply_xyz_only(temp_dir):
    """A PLY carrying only coordinates loads to a PointCloud whose xyz is [N, 3]."""
    filepath = os.path.join(temp_dir, "xyz_only.ply")
    coordinates = write_ply(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert isinstance(result, PointCloud), f"{type(result)=}"
    assert result.xyz.shape == (len(coordinates), 3), f"{result.xyz.shape=}"
    assert result.xyz.dtype == torch.float32, f"{result.xyz.dtype=}"


def test_ply_with_rgb(temp_dir):
    """RGB columns arrive as an rgb field in the dtype the file stored them in."""
    filepath = os.path.join(temp_dir, "with_rgb.ply")
    coordinates = write_ply(filepath, with_rgb=True)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.rgb.shape == (len(coordinates), 3), f"{result.rgb.shape=}"
    assert result.rgb.dtype == torch.uint8, f"{result.rgb.dtype=}"


def test_ply_extra_field_keeps_its_own_name(temp_dir):
    """A non-standard PLY column is loaded under the name the file gives it, not renamed to feat."""
    filepath = os.path.join(temp_dir, "extra_field.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert 'intensity' in result.field_names(), f"{result.field_names()=}"
    assert 'feat' not in result.field_names(), f"{result.field_names()=}"


def test_ply_name_feat_leaves_a_loaded_column_alone(temp_dir):
    """name_feat fills feat only for a column not already loaded under its own name."""
    filepath = os.path.join(temp_dir, "name_feat.ply")
    write_ply(filepath, extra_field='intensity')

    result = load_point_cloud(filepath=filepath, name_feat='intensity', device='cpu')

    assert 'intensity' in result.field_names(), f"{result.field_names()=}"
    assert 'feat' not in result.field_names(), f"{result.field_names()=}"


def test_ply_custom_element_name(temp_dir):
    """A single element named anything but vertex is found without nameInPly being given."""
    filepath = os.path.join(temp_dir, "custom_element.ply")
    coordinates = write_ply(filepath, element_name='points')

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (len(coordinates), 3), f"{result.xyz.shape=}"


def test_txt_xyz_only(temp_dir):
    """A three-column text file loads its coordinates and carries no feat."""
    filepath = os.path.join(temp_dir, "xyz_only.txt")
    table = write_txt(filepath, num_columns=3)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (len(table), 3), f"{result.xyz.shape=}"
    assert 'feat' not in result.field_names(), f"{result.field_names()=}"


def test_txt_slpccd_label_column(temp_dir):
    """In the seven-column SLPCCD layout the label column alone becomes feat."""
    filepath = os.path.join(temp_dir, "slpccd.txt")
    table = write_txt(filepath, num_columns=7)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.feat.shape == (len(table), 1), f"{result.feat.shape=}"
    expected = torch.tensor(table[:, 6:7], dtype=result.feat.dtype)
    assert torch.equal(result.feat, expected), f"{result.feat=}, {expected=}"


def test_txt_middle_width_uses_every_trailing_column(temp_dir):
    """Between four and six columns every column past the third becomes feat."""
    filepath = os.path.join(temp_dir, "middle_width.txt")
    table = write_txt(filepath, num_columns=5)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.feat.shape == (len(table), 2), f"{result.feat.shape=}"


def test_pth_tensor(temp_dir):
    """A saved torch tensor splits into coordinates and features at the third column."""
    filepath = os.path.join(temp_dir, "tensor.pth")
    write_pth(filepath, torch.rand(12, 4))

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (12, 3), f"{result.xyz.shape=}"
    assert result.feat.shape == (12, 1), f"{result.feat.shape=}"


def test_pth_ndarray(temp_dir):
    """A saved numpy array is accepted on the same terms as a tensor."""
    filepath = os.path.join(temp_dir, "ndarray.pth")
    write_pth(filepath, np.random.rand(12, 3).astype(np.float32))

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (12, 3), f"{result.xyz.shape=}"


def test_pth_rejects_a_non_array_payload(temp_dir):
    """A .pth holding anything but a tensor or an array is rejected rather than half-read."""
    filepath = os.path.join(temp_dir, "mapping.pth")
    write_pth(filepath, {'xyz': [[0.0, 0.0, 0.0]]})

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_off_xyz_only(temp_dir):
    """An OFF file's vertex block loads as coordinates."""
    filepath = os.path.join(temp_dir, "xyz_only.off")
    write_off(
        filepath,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    )

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (4, 3), f"{result.xyz.shape=}"


def test_off_keeps_only_the_leading_three_columns(temp_dir):
    """A vertex line carrying more than three numbers contributes only its coordinates."""
    filepath = os.path.join(temp_dir, "wide_vertices.off")
    write_off(
        filepath,
        [
            (0.0, 0.0, 0.0, 255.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 255.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 255.0),
        ],
    )

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.shape == (3, 3), f"{result.xyz.shape=}"
    expected = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=result.xyz.dtype
    )
    assert torch.equal(result.xyz, expected), f"{result.xyz=}, {expected=}"


def test_off_without_its_header_is_rejected(temp_dir):
    """A file whose first line is not OFF is rejected rather than parsed as vertices."""
    filepath = os.path.join(temp_dir, "malformed.off")
    write_off(filepath, [(0.0, 0.0, 0.0)], header='INVALID')

    with pytest.raises(AssertionError, match="Invalid OFF file format"):
        load_point_cloud(filepath=filepath, device='cpu')


def test_load_from_ply_returns_a_field_dict(temp_dir):
    """The PLY reader hands back a plain dict of fields, not a PointCloud, and takes no device."""
    filepath = os.path.join(temp_dir, "reader.ply")
    write_ply(filepath, with_rgb=True, extra_field='intensity')

    result = _load_from_ply(filepath=filepath)

    assert isinstance(result, dict), f"{type(result)=}"
    assert set(result.keys()) == {'xyz', 'rgb', 'intensity'}, f"{result.keys()=}"
    assert isinstance(result['xyz'], np.ndarray), f"{type(result['xyz'])=}"
    assert result['xyz'].dtype == np.float64, f"{result['xyz'].dtype=}"


def test_load_from_txt_returns_a_field_dict(temp_dir):
    """The text reader hands back a plain dict of float64 arrays."""
    filepath = os.path.join(temp_dir, "reader.txt")
    write_txt(filepath, num_columns=7)

    result = _load_from_txt(filepath=filepath)

    assert isinstance(result, dict), f"{type(result)=}"
    assert set(result.keys()) == {'xyz', 'feat'}, f"{result.keys()=}"
    assert result['xyz'].dtype == np.float64, f"{result['xyz'].dtype=}"


def test_load_from_pth_returns_a_field_dict(temp_dir):
    """The .pth reader hands back a dict in whatever form the file was saved in."""
    filepath = os.path.join(temp_dir, "reader.pth")
    write_pth(filepath, torch.rand(12, 4))

    result = _load_from_pth(filepath=filepath, device='cpu')

    assert isinstance(result, dict), f"{type(result)=}"
    assert isinstance(result['xyz'], torch.Tensor), f"{type(result['xyz'])=}"


def test_load_from_off_returns_a_field_dict(temp_dir):
    """The OFF reader hands back a dict whose xyz is already a tensor on the requested device."""
    filepath = os.path.join(temp_dir, "reader.off")
    write_off(
        filepath,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    )

    result = _load_from_off(filepath=filepath, device='cpu')

    assert isinstance(result, dict), f"{type(result)=}"
    assert result['xyz'].shape == (4, 3), f"{result['xyz'].shape=}"
    assert result['xyz'].dtype == torch.float32, f"{result['xyz'].dtype=}"
    assert result['xyz'].device.type == 'cpu', f"{result['xyz'].device=}"


def test_missing_file_is_rejected():
    """A path naming no file is rejected before any reader is chosen."""
    with pytest.raises(AssertionError, match="Point cloud file not found"):
        load_point_cloud(filepath="nonexistent.ply", device='cpu')


def test_unsupported_extension_is_rejected(temp_dir):
    """An extension no reader owns is rejected, and the file's contents are never opened."""
    filepath = os.path.join(temp_dir, "unsupported.xyz")
    with open(filepath, 'w') as handle:
        handle.write("not a point cloud at all")

    with pytest.raises(AssertionError, match="Unsupported file format"):
        load_point_cloud(filepath=filepath, device='cpu')
