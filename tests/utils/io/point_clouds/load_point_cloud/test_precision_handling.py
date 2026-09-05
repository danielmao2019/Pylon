import os
import tempfile

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud


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


def write_float64_ply(filepath: str, coordinates: np.ndarray) -> np.ndarray:
    """Write a PLY whose x, y and z are stored as float64 so precision is the reader's to lose.

    Args:
        filepath: Destination path of the PLY file.
        coordinates: [N, 3] float64 array of coordinates to store.

    Returns:
        The coordinates argument, unchanged.
    """
    rows = np.array(
        [tuple(row) for row in coordinates],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
    )
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)
    return coordinates


def test_float64_preserves_large_utm_coordinates(temp_dir):
    """Under float64 a UTM-scale coordinate comes back to within float64's own resolution."""
    filepath = os.path.join(temp_dir, "utm.ply")
    coordinates = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )
    write_float64_ply(filepath, coordinates)

    result = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)

    np.testing.assert_allclose(
        result.xyz.cpu().numpy(), coordinates, rtol=0.0, atol=1e-9
    )


def test_float32_loses_large_utm_coordinates(temp_dir):
    """Under float32 the same coordinates lose resolution, which is why the dtype argument exists."""
    filepath = os.path.join(temp_dir, "utm.ply")
    coordinates = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )
    write_float64_ply(filepath, coordinates)

    result = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float32)

    largest_error = np.abs(
        result.xyz.cpu().numpy().astype(np.float64) - coordinates
    ).max()
    assert largest_error > 1e-3, f"{largest_error=}"


def test_millimetre_offsets_survive_float64(temp_dir):
    """Two points a millimetre apart at UTM magnitude stay distinguishable under float64."""
    filepath = os.path.join(temp_dir, "millimetre.ply")
    coordinates = np.array(
        [
            [537000.000000, 4805000.000000, 350.000000],
            [537000.001000, 4805000.001000, 350.001000],
        ],
        dtype=np.float64,
    )
    write_float64_ply(filepath, coordinates)

    result = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)

    loaded = result.xyz.cpu().numpy()
    np.testing.assert_allclose(loaded[1] - loaded[0], 0.001, rtol=0.0, atol=1e-9)


def test_text_coordinates_keep_their_precision(temp_dir):
    """The text reader parses in float64, so a long decimal survives the same way."""
    filepath = os.path.join(temp_dir, "precise.txt")
    coordinates = np.array(
        [
            [537000.123456789, 4805000.987654321, 350.123456789],
            [537001.123456789, 4805001.987654321, 351.123456789],
        ],
        dtype=np.float64,
    )
    with open(filepath, 'w') as handle:
        handle.write("# header line 1\n")
        handle.write("# header line 2\n")
        for row in coordinates:
            handle.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n")

    result = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)

    np.testing.assert_allclose(
        result.xyz.cpu().numpy(), coordinates, rtol=0.0, atol=1e-9
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a cuda device")
def test_device_transfer_keeps_precision(temp_dir):
    """Moving to another device changes where the values live and not what they are."""
    filepath = os.path.join(temp_dir, "utm.ply")
    coordinates = np.array(
        [
            [537000.123456, 4805000.654321, 350.987654],
            [537001.123456, 4805001.654321, 351.987654],
        ],
        dtype=np.float64,
    )
    write_float64_ply(filepath, coordinates)

    on_cpu = load_point_cloud(filepath=filepath, device='cpu', dtype=torch.float64)
    on_cuda = load_point_cloud(filepath=filepath, device='cuda', dtype=torch.float64)

    assert torch.equal(on_cpu.xyz, on_cuda.xyz.cpu()), f"{on_cpu.xyz=}, {on_cuda.xyz=}"
