import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.random_select import RandomSelect
from data.structures.three_d.point_cloud.select import Select


def test_pointcloud_initialization() -> None:
    xyz = torch.arange(12, dtype=torch.float32).view(4, 3)
    feat = torch.arange(8, dtype=torch.float32).view(4, 2)
    pc = PointCloud(data={'xyz': xyz, 'feat': feat})
    assert pc.num_points == 4
    assert pc.field_names() == ('xyz', 'feat')
    assert torch.equal(pc.xyz, xyz)


def test_select_with_legacy_mapping_input() -> None:
    xyz = torch.randn(5, 3)
    feat = torch.randn(5, 1)
    pc = PointCloud(data={'xyz': xyz, 'feat': feat})
    out = Select(indices=[0, 3])(pc)
    assert isinstance(out, PointCloud)
    assert torch.equal(out.xyz, xyz[[0, 3]])
    assert torch.equal(out.feat, feat[[0, 3]])
    assert torch.equal(out.indices, torch.tensor([0, 3], dtype=torch.int64))


@pytest.mark.parametrize(
    "xyz_values,feat_values,indices,expected_xyz_indices,expected_feat_indices",
    [
        (
            torch.arange(12, dtype=torch.float32).view(4, 3),
            torch.arange(8, dtype=torch.float32).view(4, 2),
            [0, 2],
            [0, 2],
            [0, 2],
        ),
    ],
)
def test_select_pointcloud(
    xyz_values: torch.Tensor,
    feat_values: torch.Tensor,
    indices: list[int],
    expected_xyz_indices: list[int],
    expected_feat_indices: list[int],
) -> None:
    pc = PointCloud(data={'xyz': xyz_values, 'feat': feat_values})
    out = Select(indices=indices)(pc)
    assert torch.equal(out.xyz, xyz_values[expected_xyz_indices])
    assert torch.equal(out.feat, feat_values[expected_feat_indices])
    assert torch.equal(out.indices, torch.tensor(indices, dtype=torch.int64))


@pytest.mark.parametrize(
    "count,seed,num_points",
    [
        (3, 0, 10),
        (5, 1, 20),
    ],
)
def test_random_select_pointcloud(count: int, seed: int, num_points: int) -> None:
    xyz = torch.randn(num_points, 3)
    pc = PointCloud(xyz=xyz)
    out = RandomSelect(count=count)(pc, seed=seed)
    assert isinstance(out, PointCloud)
    assert out.num_points == min(count, num_points)
    assert out.indices.dtype == torch.int64
    assert out.indices.shape[0] == out.num_points
