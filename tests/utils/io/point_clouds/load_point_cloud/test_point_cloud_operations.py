import os
import tempfile

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
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


def test_values_survive_the_load(temp_dir):
    """The coordinates that come back are the ones that were saved."""
    filepath = os.path.join(temp_dir, "known.pth")
    saved = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    torch.save(saved, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert torch.equal(result.xyz, saved), f"{result.xyz=}, {saved=}"


def test_placed_on_the_requested_device(temp_dir):
    """Every field lands on the device the caller named."""
    filepath = os.path.join(temp_dir, "fields.pth")
    torch.save(torch.rand(12, 4), filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    for name in result.field_names():
        assert getattr(result, name).device.type == 'cpu', f"{name=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a cuda device")
def test_placed_on_cuda_when_asked(temp_dir):
    """The same placement holds for a cuda device, where one is available."""
    filepath = os.path.join(temp_dir, "fields.pth")
    torch.save(torch.rand(12, 4), filepath)

    result = load_point_cloud(filepath=filepath, device='cuda')

    assert result.xyz.device.type == 'cuda', f"{result.xyz.device=}"


def test_dtype_applies_to_xyz_only(temp_dir):
    """The dtype argument governs the coordinates and leaves every other field as stored."""
    filepath = os.path.join(temp_dir, "with_feat.pth")
    torch.save(torch.rand(12, 4, dtype=torch.float32), filepath)

    result = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)

    assert result.xyz.dtype == torch.float64, f"{result.xyz.dtype=}"
    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_segmentation_feature_column_becomes_int64(temp_dir):
    """A file whose basename carries _seg has its feature column read as labels."""
    filepath = os.path.join(temp_dir, "scene_seg.pth")
    data = torch.cat([torch.rand(12, 3), torch.randint(0, 10, (12, 1)).float()], dim=1)
    torch.save(data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.feat.dtype == torch.int64, f"{result.feat.dtype=}"


def test_feature_column_is_left_alone_without_the_seg_marker(temp_dir):
    """The same file without _seg in its basename keeps its feature column's dtype."""
    filepath = os.path.join(temp_dir, "scene.pth")
    data = torch.cat([torch.rand(12, 3), torch.randint(0, 10, (12, 1)).float()], dim=1)
    torch.save(data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_uint16_field_becomes_int32(temp_dir):
    """torch carries no uint16, so such a field arrives as the narrowest type that holds it."""
    filepath = os.path.join(temp_dir, "uint16.ply")
    rows = np.array(
        [(float(i), float(i), float(i), i * 100) for i in range(8)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u2')],
    )
    PlyData([PlyElement.describe(rows, 'vertex')], text=True).write(filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.label.dtype == torch.int32, f"{result.label.dtype=}"


def test_uppercase_seg_marker_leaves_the_feature_column_alone(temp_dir):
    """The _seg marker is matched as stored, so an uppercase _SEG basename is not a segmentation file."""
    filepath = os.path.join(temp_dir, "scene_SEG.pth")
    data = torch.cat([torch.rand(12, 3), torch.randint(0, 10, (12, 1)).float()], dim=1)
    torch.save(data, filepath)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.feat.dtype == torch.float32, f"{result.feat.dtype=}"


def test_integer_xyz_is_rejected(temp_dir):
    """An integer coordinate block is rejected rather than cast into a valid-looking float one."""
    filepath = os.path.join(temp_dir, "integer_xyz.pth")
    torch.save(torch.randint(0, 10, (12, 3)), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')


def test_windows_style_path_resolves(temp_dir):
    """A path written with backslashes names the same file as the one written with slashes."""
    filepath = os.path.join(temp_dir, "windows.pth")
    torch.save(torch.rand(12, 3), filepath)
    windows_style_path = filepath.replace('/', '\\')

    result = load_point_cloud(filepath=windows_style_path, device='cpu')

    assert isinstance(result, PointCloud), f"{type(result)=}"


def test_sizes_from_one_point_upward(temp_dir):
    """Point clouds of any row count load with their row count preserved."""
    for size in [1, 10, 1000]:
        filepath = os.path.join(temp_dir, f"size_{size}.pth")
        torch.save(torch.rand(size, 3), filepath)

        result = load_point_cloud(filepath=filepath, device='cpu')

        assert result.xyz.shape == (size, 3), f"{size=}, {result.xyz.shape=}"


def test_empty_point_cloud_is_rejected(temp_dir):
    """A file holding no points is rejected rather than yielding an empty PointCloud."""
    filepath = os.path.join(temp_dir, "empty.pth")
    torch.save(torch.empty(0, 3), filepath)

    with pytest.raises(AssertionError):
        load_point_cloud(filepath=filepath, device='cpu')
