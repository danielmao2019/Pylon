from typing import List

import numpy as np
import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.random_select import RandomSelect
from data.structures.three_d.point_cloud.select import Select


def test_a_selection_carries_the_meta_data_across() -> None:
    """The meta data travels with the fields, so a selected cloud still knows what each field's source held."""
    pc = PointCloud(
        data={
            'xyz': np.zeros((5, 3), dtype=np.float32),
            'intensity': np.arange(5, dtype=np.uint16),
        },
        meta_data={
            'xyz': {'dtype': 'float32', 'layout': ('x', 'y', 'z')},
            'intensity': {'dtype': 'uint16', 'layout': ('intensity',)},
        },
    )

    select = Select(indices=[0, 3])
    out = select(pc)

    assert out.meta_data['intensity']['dtype'] == 'uint16'
    assert out.meta_data['xyz']['layout'] == ('x', 'y', 'z')


def test_the_indices_a_selection_makes_are_named_by_no_meta_data() -> None:
    """A selection inherits the meta data whole rather than building one, so the indices field it makes itself sits outside the meta data and carries only the dtype it means."""
    pc = PointCloud(xyz=torch.randn(5, 3, dtype=torch.float32))

    select = Select(indices=[1, 2])
    out = select(pc)

    assert 'indices' not in out.meta_data
    # its indices tensor is int64, which is what a field outside the meta data means
    assert out.indices.dtype == torch.int64


def test_pointcloud_initialization() -> None:
    """A point cloud built from a field dict reports its point count, its field names in coordinates-first order, and its coordinates."""
    xyz = torch.arange(12, dtype=torch.float32).view(4, 3)
    feat = torch.arange(8, dtype=torch.float32).view(4, 2)

    pc = PointCloud(data={'xyz': xyz, 'feat': feat})

    assert pc.num_points == 4
    assert pc.field_names() == ('xyz', 'feat')
    assert torch.equal(pc.xyz, xyz)


def test_select_takes_a_plain_index_list() -> None:
    """Selecting by a plain index list carries every field down and adds the taken indices as a field."""
    xyz = torch.randn(5, 3, dtype=torch.float32)
    feat = torch.randn(5, 1, dtype=torch.float32)
    pc = PointCloud(data={'xyz': xyz, 'feat': feat})

    select = Select(indices=[0, 3])
    out = select(pc)

    assert isinstance(out, PointCloud)
    assert torch.equal(out.xyz, xyz[[0, 3]])
    assert torch.equal(out.feat, feat[[0, 3]])
    assert torch.equal(out.indices, torch.tensor([[0], [3]], dtype=torch.int64))


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
    indices: List[int],
    expected_xyz_indices: List[int],
    expected_feat_indices: List[int],
) -> None:
    """Each parametrized selection takes exactly the named rows of every field and adds the indices it took as a field."""
    pc = PointCloud(data={'xyz': xyz_values, 'feat': feat_values})

    select = Select(indices=indices)
    out = select(pc)

    assert torch.equal(out.xyz, xyz_values[expected_xyz_indices])
    assert torch.equal(out.feat, feat_values[expected_feat_indices])
    assert torch.equal(
        out.indices, torch.tensor([[index] for index in indices], dtype=torch.int64)
    )


@pytest.mark.parametrize(
    "count,seed,num_points",
    [
        (3, 0, 10),
        (5, 1, 20),
    ],
)
def test_random_select_pointcloud(count: int, seed: int, num_points: int) -> None:
    """A seeded random selection of a fixed count hands back that many points and an int64 index field of the same length."""
    pc = PointCloud(xyz=torch.randn(num_points, 3, dtype=torch.float32))

    random_select = RandomSelect(count=count)
    out = random_select(pc, seed=seed)

    assert isinstance(out, PointCloud)
    assert out.num_points == min(count, num_points)
    assert out.indices.dtype == torch.int64
    assert out.indices.shape[0] == out.num_points
