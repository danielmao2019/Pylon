"""
Tests for the SiameseKPConvCollator.

This module contains tests for verifying the functionality of the SiameseKPConvCollator
for 3D point cloud change detection.
"""

import pytest
import torch

from data.collators.siamese_kpconv_collator import SiameseKPConvCollator
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def _build_point_cloud(num_points: int) -> PointCloud:
    """Create a synthetic PointCloud with XYZ + 3 feature dims and zero batch indices."""
    tensor = torch.rand(num_points, 6)
    return PointCloud(
        xyz=tensor[:, :3],
        data={
            'feat': tensor[:, 3:],
            'batch': torch.zeros(num_points, dtype=torch.long),
        },
    )


def _make_sample(num_points: int, sample_idx: int) -> dict:
    """Build a datapoint sample with two point clouds and metadata."""
    return {
        "inputs": {
            "pc_0": _build_point_cloud(num_points),
            "pc_1": _build_point_cloud(num_points),
        },
        "labels": {"change_map": torch.randint(0, 2, (num_points,), dtype=torch.long)},
        "meta_info": {"sample_idx": sample_idx, "filename": f"sample_{sample_idx}.ply"},
    }


@pytest.mark.parametrize(
    "sample_sizes, expected_batch_size",
    [
        ([10, 10], 20),
        ([5, 7, 3], 15),
    ],
)
def test_siamese_kpconv_collator(sample_sizes, expected_batch_size):
    """
    Test the SiameseKPConvCollator with different input scenarios.

    Args:
        samples: List of input samples
        expected_batch_size: Expected total number of points after batching
    """
    torch.manual_seed(0)
    samples = [_make_sample(size, idx) for idx, size in enumerate(sample_sizes)]

    # Apply the collator
    collator = SiameseKPConvCollator()
    batch = collator(samples)

    # Check batch structure
    assert set(batch.keys()) == {'inputs', 'labels', 'meta_info'}
    assert set(batch['inputs'].keys()) == {'pc_0', 'pc_1'}
    assert set(batch['labels'].keys()) == {'change'}

    # Check that the point clouds were batched correctly
    for pc_key in ['pc_0', 'pc_1']:
        pc = batch['inputs'][pc_key]
        assert isinstance(pc, PointCloud)

        # Verify tensor shapes
        assert pc.num_points == expected_batch_size
        assert pc.xyz.shape[1] == 3  # xyz coordinates
        assert pc.feat.shape[0] == expected_batch_size
        assert pc.feat.shape[1] == 3  # features
        assert pc.batch.shape[0] == expected_batch_size

        # Verify batch indices
        batch_indices = pc.batch
        for i, sample in enumerate(samples):
            sample_size = sample['inputs'][pc_key].shape[0]
            if i == 0:
                start_idx = 0
            else:
                start_idx = sum(s['inputs'][pc_key].shape[0] for s in samples[:i])

            end_idx = start_idx + sample_size
            expected_indices = torch.full((sample_size,), i, dtype=torch.long)
            assert torch.all(
                batch_indices[start_idx:end_idx] == expected_indices
            ), f"Batch indices for sample {i} don't match expected values"

    # Check change map
    assert batch['labels']['change'].shape[0] == expected_batch_size

    # Check consistency between batched pc_1 and change map
    assert batch['inputs']['pc_1'].num_points == batch['labels']['change'].shape[0]
