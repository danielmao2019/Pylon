import torch
import pytest

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


def test_select_basic_list() -> None:
    xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    classification = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb, 'classification': classification})

    select = Select([0, 2, 4])
    result = select(pc)

    expected_xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    expected_rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    expected_cls = torch.tensor([0, 2, 1], dtype=torch.long)

    assert torch.allclose(result.xyz, expected_xyz)
    assert torch.allclose(result.rgb, expected_rgb)
    assert torch.equal(result.classification, expected_cls)
    assert torch.equal(result.indices, torch.tensor([0, 2, 4], dtype=torch.int64))


def test_select_basic_tensor() -> None:
    xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb})

    indices_tensor = torch.tensor([1, 2], dtype=torch.int64, device=pc.xyz.device)
    select = Select(indices_tensor)
    result = select(pc)

    expected_xyz = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result.xyz, expected_xyz)
    assert torch.equal(result.indices, indices_tensor)


def test_select_empty_indices() -> None:
    xyz = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    classification = torch.tensor([0, 1], dtype=torch.long)
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb, 'classification': classification})

    select = Select([])
    with pytest.raises(AssertionError):
        _ = select(pc)


def test_select_single_point() -> None:
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    rgb = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb})

    select = Select([1])
    result = select(pc)

    expected_xyz = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(result.xyz, expected_xyz)
    assert result.rgb.shape == (1, 3)
    assert torch.equal(result.indices, torch.tensor([1], dtype=torch.int64))


def test_select_out_of_order() -> None:
    xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select([3, 0, 2])
    result = select(pc)

    expected_xyz = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result.xyz, expected_xyz)
    assert torch.equal(result.indices, torch.tensor([3, 0, 2], dtype=torch.int64))


def test_select_duplicate_indices() -> None:
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select([1, 1, 2, 1])
    result = select(pc)

    expected_xyz = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(result.xyz, expected_xyz)
    assert torch.equal(result.indices, torch.tensor([1, 1, 2, 1], dtype=torch.int64))
