import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.random_select import RandomSelect


def test_random_select_percentage_basic() -> None:
    """A percentage selection keeps that fraction of the points and carries the color field and the index field down with it."""
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
    # a selection indexes fields down, it does not re-derive what they came from
    assert result.meta_data['rgb'] == pc.meta_data['rgb']


def test_random_select_count_basic() -> None:
    """A count selection of fewer points than the cloud carries hands back exactly that many."""
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
    """Two selections under the same seed draw the very same points in the very same order."""
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
    """A count larger than the cloud is capped at the number of points there are."""
    xyz = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=5)
    result = random_select(pc, seed=42)

    assert result.num_points == 2
    assert result.indices.shape[0] == 2


def test_random_select_takes_a_quarter_of_twenty() -> None:
    """A quarter of a twenty-point cloud is five points."""
    xyz = torch.tensor([[float(i), 0.0, 0.0] for i in range(20)], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(percentage=0.25)
    result = random_select(pc, seed=42)

    assert result.num_points == int(20 * 0.25)


def test_random_select_takes_ten_of_twenty() -> None:
    """A count of ten out of a twenty-point cloud is ten points."""
    xyz = torch.tensor([[float(i), 0.0, 0.0] for i in range(20)], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=10)
    result = random_select(pc, seed=42)

    assert result.num_points == 10


def test_random_select_takes_exactly_one_sizing_mode() -> None:
    """The two modes size the selection differently, so naming both or neither leaves it undefined rather than defaulting to one."""
    with pytest.raises(AssertionError):
        RandomSelect()

    with pytest.raises(AssertionError):
        RandomSelect(percentage=0.5, count=3)


def test_random_select_refuses_a_percentage_outside_its_range() -> None:
    """A percentage at or below zero selects no points and one above one selects more than there are, so both are refused where the mode is chosen."""
    with pytest.raises(AssertionError):
        RandomSelect(percentage=0.0)

    with pytest.raises(AssertionError):
        RandomSelect(percentage=1.5)


def test_random_select_refuses_a_count_that_is_not_positive() -> None:
    """A selection of no points is not a point cloud, so the count is refused at construction rather than producing one downstream."""
    with pytest.raises(AssertionError):
        RandomSelect(count=0)


def test_random_select_takes_exactly_one_source_of_randomness() -> None:
    """A seed and a generator are two ways to fix the same draw, so naming both or neither leaves the draw undefined."""
    xyz = torch.tensor([[float(i), 0.0, 0.0] for i in range(8)], dtype=torch.float64)
    pc = PointCloud(xyz=xyz)

    random_select = RandomSelect(count=3)

    with pytest.raises(AssertionError):
        random_select(pc)

    with pytest.raises(AssertionError):
        random_select(pc, seed=42, generator=torch.Generator(device=pc.device))
