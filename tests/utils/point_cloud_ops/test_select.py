import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


def test_select_basic_list() -> None:
    """An index list takes those rows of coordinates, colors and a classification field alike, and adds them as the indices field."""
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
    classification = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb, 'classification': classification})

    select = Select([0, 2, 4])
    result = select(pc)

    assert torch.equal(result.xyz, pc.xyz[[0, 2, 4]])
    assert torch.equal(result.rgb, pc.rgb[[0, 2, 4]])
    assert torch.equal(result.classification, pc.classification[[0, 2, 4]])
    assert torch.equal(result.indices, torch.tensor([[0], [2], [4]], dtype=torch.int64))


def test_select_basic_tensor() -> None:
    """An int64 index tensor on the point cloud's device selects exactly as an index list does."""
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

    indices_tensor = torch.tensor([1, 2], dtype=torch.int64, device=pc.device)
    select = Select(indices_tensor)
    result = select(pc)

    assert torch.equal(result.xyz, pc.xyz[[1, 2]])
    assert torch.equal(
        result.indices, torch.tensor([[1], [2]], dtype=torch.int64, device=pc.device)
    )


def test_select_empty_indices() -> None:
    """Selecting no points at all is refused, because a point cloud carries at least one point."""
    xyz = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    classification = torch.tensor([0, 1], dtype=torch.int64)
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb, 'classification': classification})

    select = Select([])

    with pytest.raises(AssertionError):
        select(pc)


def test_select_single_point() -> None:
    """A one-entry selection hands back a one-point cloud whose color field keeps its trailing three columns."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    rgb = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz, 'rgb': rgb})

    select = Select([1])
    result = select(pc)

    assert torch.equal(result.xyz, pc.xyz[[1]])
    assert result.rgb.shape == (1, 3)
    assert torch.equal(result.indices, torch.tensor([[1]], dtype=torch.int64))


def test_select_out_of_order() -> None:
    """The selected rows come back in the order the indices name."""
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

    assert torch.equal(result.xyz, pc.xyz[[3, 0, 2]])
    assert torch.equal(result.indices, torch.tensor([[3], [0], [2]], dtype=torch.int64))


def test_select_refuses_an_index_past_the_last_point() -> None:
    """An index naming a point the cloud does not have is refused at the door rather than reaching torch and coming back as an opaque indexing error."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select([0, 5])

    with pytest.raises(AssertionError):
        select(pc)


def test_select_refuses_a_negative_index() -> None:
    """A negative index would silently select from the far end, so it is refused rather than wrapping."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select([-1])

    with pytest.raises(AssertionError):
        select(pc)


def test_select_refuses_a_non_int64_index_tensor() -> None:
    """The int64 requirement is asserted on the indices the selection was constructed with, not only on an indices field the cloud already carried."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select(torch.tensor([0, 2], dtype=torch.int32))

    with pytest.raises(AssertionError):
        select(pc)


def test_select_refuses_an_index_tensor_on_another_device() -> None:
    """A selection materializes a list onto the cloud's device but never moves a tensor it was handed, so a mismatch aborts instead of transferring silently."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
        device='cpu',
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select(torch.tensor([0, 2], dtype=torch.int64, device='cuda'))

    with pytest.raises(AssertionError):
        select(pc)


def test_select_duplicate_indices() -> None:
    """A repeated index takes the same row again, so the selection may be longer than the point cloud it came from."""
    xyz = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    pc = PointCloud(data={'xyz': xyz})

    select = Select([1, 1, 2, 1])
    result = select(pc)

    assert torch.equal(result.xyz, pc.xyz[[1, 1, 2, 1]])
    assert torch.equal(
        result.indices, torch.tensor([[1], [1], [2], [1]], dtype=torch.int64)
    )
