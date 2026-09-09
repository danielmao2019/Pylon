import os
import tempfile

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud

# coordinates of UTM magnitude, whose integer part alone eats most of float32's significand
UTM_COORDINATES = np.array(
    [
        [500000.123456789, 4000000.987654321, 123.456789012],
        [500001.234567891, 4000001.876543219, 124.567890123],
        [500002.345678912, 4000002.765432198, 125.678901234],
        [500003.456789123, 4000003.654321987, 126.789012345],
    ],
    dtype=np.float64,
)


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


def test_an_f8_ply_gives_float64_coordinates(temp_dir):
    """A reader neither widens nor narrows, so an f8 file's UTM-scale coordinates come back to within float64's own resolution without anything being asked for."""
    filepath = os.path.join(temp_dir, "float64.ply")
    write_float64_ply(filepath, UTM_COORDINATES)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.dtype == torch.float64, f"{result.xyz.dtype=}"
    assert np.allclose(
        result.xyz.numpy(), UTM_COORDINATES, atol=1e-9
    ), f"{result.xyz.numpy()=}, {UTM_COORDINATES=}"


def test_an_f4_ply_gives_float32_coordinates_and_their_loss(temp_dir):
    """The same coordinates stored as f4 come back as float32 and carry that width's own loss, because the reader keeps the file's dtype rather than widening to cover it."""
    filepath = os.path.join(temp_dir, "float32.ply")
    write_float32_ply(filepath, UTM_COORDINATES)

    result = load_point_cloud(filepath=filepath, device='cpu')

    assert result.xyz.dtype == torch.float32, f"{result.xyz.dtype=}"
    largest_error = float(
        np.abs(result.xyz.numpy().astype(np.float64) - UTM_COORDINATES).max()
    )
    assert largest_error > 1e-3, f"{largest_error=}"


def test_millimetre_offsets_survive_float64(temp_dir):
    """Two points a millimetre apart at UTM magnitude stay distinguishable under float64."""
    filepath = os.path.join(temp_dir, "millimetre.ply")
    coordinates = np.array(
        [
            [500000.000000000, 4000000.000000000, 100.000000000],
            [500000.001000000, 4000000.000000000, 100.000000000],
        ],
        dtype=np.float64,
    )
    write_float64_ply(filepath, coordinates)

    result = load_point_cloud(filepath=filepath, device='cpu')

    gap = float(result.xyz[1, 0] - result.xyz[0, 0])
    assert abs(gap - 0.001) < 1e-9, f"{gap=}"


def test_text_coordinates_keep_their_precision(temp_dir):
    """The text reader parses in float64, so a long decimal survives the same way."""
    filepath = os.path.join(temp_dir, "precise.txt")
    with open(filepath, 'w') as f:
        f.write("header line one\n")
        f.write("header line two\n")
        for row in UTM_COORDINATES:
            f.write(" ".join(f"{value:.9f}" for value in row) + "\n")

    result = load_point_cloud(
        filepath=filepath,
        meta_data={'xyz': {'layout': ('0', '1', '2')}},
        device='cpu',
    )

    assert np.allclose(
        result.xyz.numpy(), UTM_COORDINATES, atol=1e-9
    ), f"{result.xyz.numpy()=}, {UTM_COORDINATES=}"


def test_device_transfer_keeps_precision(temp_dir):
    """Moving to another device changes where the values live and not what they are."""
    if not torch.cuda.is_available():
        pytest.skip("no cuda device is available")
    filepath = os.path.join(temp_dir, "transfer.ply")
    write_float64_ply(filepath, UTM_COORDINATES)

    on_cpu = load_point_cloud(filepath=filepath, device='cpu')
    on_cuda = load_point_cloud(filepath=filepath, device='cuda')

    assert torch.equal(on_cpu.xyz, on_cuda.xyz.cpu()), f"{on_cpu.xyz=}, {on_cuda.xyz=}"


def write_float64_ply(filepath: str, coordinates: np.ndarray) -> None:
    """Writes a PLY whose x, y and z are stored as f8, so the width the reader hands back is the file's own.

    Args:
        filepath: Destination path of the PLY file, as a str.
        coordinates: The coordinates to store, as a [N, 3] float64 numpy array.

    Returns:
        None.
    """
    rows = np.array(
        [tuple(row) for row in coordinates],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
    )
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)


def write_float32_ply(filepath: str, coordinates: np.ndarray) -> None:
    """Writes the same coordinates as f4, so the same reader hands back the narrower width.

    Args:
        filepath: Destination path of the PLY file, as a str.
        coordinates: The coordinates to store, as a [N, 3] float64 numpy array narrowed onto f4 by the write.

    Returns:
        None.
    """
    rows = np.array(
        [tuple(row) for row in coordinates],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    PlyData([PlyElement.describe(rows, 'vertex')]).write(filepath)
