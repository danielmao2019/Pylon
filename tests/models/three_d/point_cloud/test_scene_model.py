import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from models.three_d.point_cloud.scene_model import PointCloudSceneModel


def test_a_self_describing_file_opens_with_nothing_supplied(tmp_path) -> None:
    """A single-element ply names its own columns, so a scene model opens it with no meta data of any kind."""
    filepath = str(tmp_path / "cloud.ply")
    write_ply(filepath=filepath)

    resolved_path = PointCloudSceneModel.parse_scene_path(filepath)

    model = PointCloudSceneModel(scene_path=resolved_path, device=torch.device('cpu'))
    assert (
        'xyz' in model.model.field_names()
    ), f"a self-describing ply opens into a cloud carrying coordinates: filepath={filepath}, fields={model.model.field_names()}"


def test_a_format_that_names_none_of_its_columns_is_refused_at_the_path(
    tmp_path,
) -> None:
    """A .pth defines no layout and nothing here can state one, so it is refused where the path is named rather than aborting inside the loader."""
    filepath = str(tmp_path / "cloud.pth")
    torch.save(torch.zeros(size=(8, 3), dtype=torch.float32), filepath)

    with pytest.raises(AssertionError):
        PointCloudSceneModel.parse_scene_path(filepath)


def test_a_double_precision_file_keeps_its_width_into_the_display(tmp_path) -> None:
    """No load forces a coordinate width here either, so an f8 file reaches the display as float64."""
    filepath = str(tmp_path / "cloud.ply")
    write_float64_ply(filepath=filepath)

    model = PointCloudSceneModel(scene_path=filepath, device=torch.device('cpu'))

    positions = model.extract_positions()
    assert (
        positions.dtype == torch.float64
    ), f"an f8 file reaches the display at its own width: filepath={filepath}, positions.dtype={positions.dtype}"


def write_ply(filepath: str, num_points: int = 8) -> None:
    """Writes a single-element PLY with f4 coordinates, since this suite needs a file that names its own columns.

    Args:
        filepath: The destination path of the .ply file to write, as a str.
        num_points: The number of vertex rows to write, as an int.

    Returns:
        None.
    """
    vertices = np.array(
        [
            (float(index), float(index) + 1.0, float(index) + 2.0)
            for index in range(num_points)
        ],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)


def write_float64_ply(filepath: str, num_points: int = 8) -> None:
    """Writes the same file with f8 coordinates, so the width reaching the display is the file's own.

    Args:
        filepath: The destination path of the .ply file to write, as a str.
        num_points: The number of vertex rows to write, as an int.

    Returns:
        None.
    """
    vertices = np.array(
        [
            (float(index), float(index) + 1.0, float(index) + 2.0)
            for index in range(num_points)
        ],
        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')],
    )
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
