"""
Tests for the SiameseKPConvCollator.

This module contains tests for verifying the functionality of the SiameseKPConvCollator
for 3D point cloud change detection.
"""
import pytest
import torch

from data.collators.siamese_kpconv_collator import SiameseKPConvCollator


@pytest.mark.parametrize(
    "samples, expected_batch_size",
    [
        # Test case 1: Two samples with 10 points each
        (
            [
                {
                    "inputs": {
                        "pc_0": torch.rand(10, 6),  # 10 points, 3 position + 3 features
                        "pc_1": torch.rand(10, 6)
                    },
                    "labels": {
                        "change_map": torch.randint(0, 2, (10,), dtype=torch.long)
                    },
                    "meta_info": {
                        "sample_idx": 0,
                        "filename": "sample_0.ply"
                    }
                },
                {
                    "inputs": {
                        "pc_0": torch.rand(10, 6),
                        "pc_1": torch.rand(10, 6)
                    },
                    "labels": {
                        "change_map": torch.randint(0, 2, (10,), dtype=torch.long)
                    },
                    "meta_info": {
                        "sample_idx": 1,
                        "filename": "sample_1.ply"
                    }
                }
            ],
            20  # Expected 20 points total (10 + 10)
        ),
        
        # Test case 2: Three samples with different number of points
        (
            [
                {
                    "inputs": {
                        "pc_0": torch.rand(5, 6),  # 5 points
                        "pc_1": torch.rand(5, 6)
                    },
                    "labels": {
                        "change_map": torch.randint(0, 2, (5,), dtype=torch.long)
                    },
                    "meta_info": {
                        "sample_idx": 0,
                        "filename": "sample_0.ply"
                    }
                },
                {
                    "inputs": {
                        "pc_0": torch.rand(7, 6),  # 7 points
                        "pc_1": torch.rand(7, 6)
                    },
                    "labels": {
                        "change_map": torch.randint(0, 2, (7,), dtype=torch.long)
                    },
                    "meta_info": {
                        "sample_idx": 1,
                        "filename": "sample_1.ply"
                    }
                },
                {
                    "inputs": {
                        "pc_0": torch.rand(3, 6),  # 3 points
                        "pc_1": torch.rand(3, 6)
                    },
                    "labels": {
                        "change_map": torch.randint(0, 2, (3,), dtype=torch.long)
                    },
                    "meta_info": {
                        "sample_idx": 2,
                        "filename": "sample_2.ply"
                    }
                }
            ],
            15  # Expected 15 points total (5 + 7 + 3)
        )
    ]
)
def test_siamese_kpconv_collator(samples, expected_batch_size):
    """
    Test the SiameseKPConvCollator with different input scenarios.
    
    Args:
        samples: List of input samples
        expected_batch_size: Expected total number of points after batching
    """
    # Apply the collator
    collator = SiameseKPConvCollator()
    batch = collator(samples)
    
    # Check batch structure
    assert set(batch.keys()) == {'inputs', 'labels', 'meta_info'}
    assert set(batch['inputs'].keys()) == {'pc_0', 'pc_1'}
    assert set(batch['labels'].keys()) == {'change'}
    
    # Check that the point clouds were batched correctly
    for pc_key in ['pc_0', 'pc_1']:
        # Verify tensor keys
        assert set(batch['inputs'][pc_key].keys()) == {'pos', 'x', 'batch'}
        
        # Verify tensor shapes
        assert batch['inputs'][pc_key]['pos'].shape[0] == expected_batch_size
        assert batch['inputs'][pc_key]['pos'].shape[1] == 3  # xyz coordinates
        assert batch['inputs'][pc_key]['x'].shape[0] == expected_batch_size
        assert batch['inputs'][pc_key]['x'].shape[1] == 3  # features
        assert batch['inputs'][pc_key]['batch'].shape[0] == expected_batch_size
        
        # Verify batch indices
        batch_indices = batch['inputs'][pc_key]['batch']
        for i, sample in enumerate(samples):
            sample_size = sample['inputs'][pc_key].shape[0]
            if i == 0:
                start_idx = 0
            else:
                start_idx = sum(s['inputs'][pc_key].shape[0] for s in samples[:i])
                
            end_idx = start_idx + sample_size
            expected_indices = torch.full((sample_size,), i, dtype=torch.long)
            assert torch.all(batch_indices[start_idx:end_idx] == expected_indices), \
                f"Batch indices for sample {i} don't match expected values"
    
    # Check change map
    assert batch['labels']['change'].shape[0] == expected_batch_size
    
    # Check consistency between batched pc_1 and change map
    assert batch['inputs']['pc_1']['pos'].shape[0] == batch['labels']['change'].shape[0]
