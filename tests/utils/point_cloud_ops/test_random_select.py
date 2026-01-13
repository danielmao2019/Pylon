import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.random_select import RandomSelect


def test_random_select_percentage_basic() -> None:
    xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb})

    random_select = RandomSelect(percentage=0.5)
    result = random_select(pc, seed=42)

    expected_count = int(4 * 0.5)
    assert result.num_points == expected_count
    assert result.rgb.shape[0] == expected_count
    assert result.indices.shape[0] == expected_count
    assert result.indices.dtype == torch.int64


def test_random_select_count_basic() -> None:
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
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=3)
    result = random_select(pc, seed=42)

    assert result.num_points == 3
    assert result.indices.shape[0] == 3


def test_random_select_deterministic_with_seed() -> None:
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(percentage=0.5)
    result1 = random_select(pc, seed=42)
    result2 = random_select(pc, seed=42)

    assert torch.equal(result1.xyz, result2.xyz)
    assert torch.equal(result1.indices, result2.indices)


def test_random_select_count_exceeds_points() -> None:
    xyz = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=5)
    result = random_select(pc, seed=42)
    assert result.num_points == 2
    assert result.indices.shape[0] == 2


def test_random_select_percentage_range() -> None:
    xyz = torch.tensor([[float(i), 0.0, 0.0] for i in range(20)], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(percentage=0.25)
    result = random_select(pc, seed=42)
    assert result.num_points == int(20 * 0.25)


def test_random_select_count_range() -> None:
    xyz = torch.tensor([[float(i), 0.0, 0.0] for i in range(20)], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=10)
    result = random_select(pc, seed=42)
    assert result.num_points == 10
