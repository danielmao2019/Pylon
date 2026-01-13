import pytest
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from utils.input_checks.check_point_cloud import check_point_cloud_segmentation


def test_point_cloud_keys_and_access() -> None:
    xyz = torch.randn(4, 3, dtype=torch.float32)
    pc = PointCloud(xyz=xyz)

    assert pc.num_points == 4
    assert pc.device == xyz.device


def test_setitem_validation() -> None:
    pc = PointCloud(xyz=torch.randn(5, 3, dtype=torch.float32))
    with pytest.raises(AssertionError):
        pc.feat = torch.randn(4, 2, dtype=torch.float32)

    pc.feat = torch.randn(5, 2, dtype=torch.float32)
    assert torch.equal(pc.feat, pc.feat)


def test_missing_field_access() -> None:
    pc = PointCloud(xyz=torch.randn(3, 3, dtype=torch.float32))
    with pytest.raises(AttributeError):
        _ = pc.feat


def test_point_cloud_requires_xyz() -> None:
    with pytest.raises(AssertionError):
        PointCloud(data={'feat': torch.randn(4, 1, dtype=torch.float32)})


def test_point_cloud_rejects_nan_xyz() -> None:
    with pytest.raises(AssertionError, match="xyz tensor contains NaN"):
        PointCloud(xyz=torch.tensor([[float('nan'), 0.0, 0.0]], dtype=torch.float32))


def test_point_cloud_length_mismatch_on_assignment() -> None:
    pc = PointCloud(xyz=torch.randn(5, 3, dtype=torch.float32))
    with pytest.raises(AssertionError):
        pc.feat = torch.randn(4, 2, dtype=torch.float32)


def test_reserved_attribute_assignment_rejected() -> None:
    pc = PointCloud(xyz=torch.randn(3, 3, dtype=torch.float32))
    with pytest.raises(AssertionError):
        pc.device = torch.randn(3, 3, dtype=torch.float32)


def test_non_string_keys_rejected() -> None:
    xyz = torch.randn(4, 3, dtype=torch.float32)
    with pytest.raises(AssertionError):
        PointCloud(data={'xyz': xyz, 1: xyz})


def test_point_cloud_segmentation_validation() -> None:
    logits = torch.randn(6, 4, dtype=torch.float32)
    labels = torch.randint(low=0, high=4, size=(6,), dtype=torch.int64)
    validated_logits, validated_labels = check_point_cloud_segmentation(
        y_pred=logits, y_true=labels,
    )
    assert validated_logits is logits
    assert validated_labels is labels


def test_point_cloud_segmentation_validation_errors() -> None:
    logits = torch.randn(5, 3, dtype=torch.float32)
    labels = torch.randint(low=0, high=3, size=(4,), dtype=torch.int64)
    with pytest.raises(AssertionError):
        check_point_cloud_segmentation(y_pred=logits, y_true=labels)
