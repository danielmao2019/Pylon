"""Tests for buffer_stack functionality with point cloud data structures."""

from typing import Dict, Any, List
import pytest
import torch
import numpy as np

from utils.ops.dict_as_tensor import buffer_stack


def test_buffer_stack_point_cloud_registration():
    """Test buffer_stack with point cloud registration data structures."""
    # Simulate point cloud registration input format
    sample1 = {
        'src_points': torch.randn(1024, 3, dtype=torch.float32),
        'tgt_points': torch.randn(1024, 3, dtype=torch.float32),
        'correspondences': torch.randint(0, 1024, (2, 512), dtype=torch.long),
        'transform': torch.eye(4, dtype=torch.float32)
    }
    
    sample2 = {
        'src_points': torch.randn(1024, 3, dtype=torch.float32),
        'tgt_points': torch.randn(1024, 3, dtype=torch.float32),
        'correspondences': torch.randint(0, 1024, (2, 512), dtype=torch.long),
        'transform': torch.eye(4, dtype=torch.float32)
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check that all keys are preserved
    expected_keys = {'src_points', 'tgt_points', 'correspondences', 'transform'}
    assert set(result.keys()) == expected_keys
    
    # Check stacking results (buffer_stack stacks along last dimension)
    assert result['src_points'].shape == (1024, 3, 2)  # Stacked along last dimension
    assert result['tgt_points'].shape == (1024, 3, 2)
    assert result['correspondences'].shape == (2, 512, 2)
    assert result['transform'].shape == (4, 4, 2)
    
    # Check data types are preserved
    assert result['src_points'].dtype == torch.float32
    assert result['tgt_points'].dtype == torch.float32
    assert result['correspondences'].dtype == torch.long
    assert result['transform'].dtype == torch.float32


def test_buffer_stack_nested_point_cloud_features():
    """Test buffer_stack with nested point cloud feature structures."""
    # Simulate nested feature structure
    sample1 = {
        'point_features': {
            'geometric': {
                'positions': torch.randn(512, 3, dtype=torch.float32),
                'normals': torch.randn(512, 3, dtype=torch.float32),
                'curvatures': torch.randn(512, 1, dtype=torch.float32)
            },
            'learned': {
                'embeddings': torch.randn(512, 128, dtype=torch.float32),
                'attention_weights': torch.randn(512, 8, dtype=torch.float32)
            }
        },
        'global_features': {
            'shape_context': torch.randn(256, dtype=torch.float32),
            'pose_estimate': torch.randn(6, dtype=torch.float32)  # 6DOF pose
        }
    }
    
    sample2 = {
        'point_features': {
            'geometric': {
                'positions': torch.randn(512, 3, dtype=torch.float32),
                'normals': torch.randn(512, 3, dtype=torch.float32),
                'curvatures': torch.randn(512, 1, dtype=torch.float32)
            },
            'learned': {
                'embeddings': torch.randn(512, 128, dtype=torch.float32),
                'attention_weights': torch.randn(512, 8, dtype=torch.float32)
            }
        },
        'global_features': {
            'shape_context': torch.randn(256, dtype=torch.float32),
            'pose_estimate': torch.randn(6, dtype=torch.float32)
        }
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check nested structure preservation
    assert 'point_features' in result
    assert 'global_features' in result
    assert 'geometric' in result['point_features']
    assert 'learned' in result['point_features']
    
    # Check deeply nested tensors
    assert result['point_features']['geometric']['positions'].shape == (512, 3, 2)
    assert result['point_features']['geometric']['normals'].shape == (512, 3, 2)
    assert result['point_features']['geometric']['curvatures'].shape == (512, 1, 2)
    assert result['point_features']['learned']['embeddings'].shape == (512, 128, 2)
    assert result['point_features']['learned']['attention_weights'].shape == (512, 8, 2)
    
    # Check global features
    assert result['global_features']['shape_context'].shape == (256, 2)
    assert result['global_features']['pose_estimate'].shape == (6, 2)


def test_buffer_stack_variable_point_counts():
    """Test buffer_stack error handling with variable point counts (should fail)."""
    # Create samples with different point counts (this should fail)
    sample1 = {
        'points': torch.randn(512, 3, dtype=torch.float32),
        'features': torch.randn(512, 64, dtype=torch.float32)
    }
    
    sample2 = {
        'points': torch.randn(1024, 3, dtype=torch.float32),  # Different point count
        'features': torch.randn(1024, 64, dtype=torch.float32)
    }
    
    # This should raise a runtime error due to inconsistent shapes
    with pytest.raises(RuntimeError, match="stack expects each tensor to be equal size"):
        buffer_stack([sample1, sample2])


def test_buffer_stack_transformation_matrices():
    """Test buffer_stack with transformation matrices commonly used in point cloud registration."""
    # 4x4 transformation matrices
    transform1 = torch.eye(4, dtype=torch.float32)
    transform1[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation
    
    transform2 = torch.eye(4, dtype=torch.float32)
    transform2[:3, 3] = torch.tensor([4.0, 5.0, 6.0])
    
    transform3 = torch.eye(4, dtype=torch.float32)
    transform3[:3, 3] = torch.tensor([7.0, 8.0, 9.0])
    
    result = buffer_stack([transform1, transform2, transform3])
    
    # Check shape preservation (stacked along last dimension)
    assert result.shape == (4, 4, 3)  # Stacked 3 matrices along last dimension
    
    # For 4x4 matrices, buffer_stack should create a new dimension along the last axis
    # Check that the transformations are correctly stacked
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_buffer_stack_correspondence_data():
    """Test buffer_stack with correspondence data structures."""
    # Correspondence indices and scores
    sample1 = {
        'correspondences': {
            'indices': torch.randint(0, 1000, (2, 256), dtype=torch.long),  # [src_idx, tgt_idx] pairs
            'scores': torch.rand(256, dtype=torch.float32),
            'valid_mask': torch.rand(256) > 0.3  # Boolean mask
        },
        'matching_matrix': torch.rand(100, 100, dtype=torch.float32)  # Soft matching
    }
    
    sample2 = {
        'correspondences': {
            'indices': torch.randint(0, 1000, (2, 256), dtype=torch.long),
            'scores': torch.rand(256, dtype=torch.float32),
            'valid_mask': torch.rand(256) > 0.3
        },
        'matching_matrix': torch.rand(100, 100, dtype=torch.float32)
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check structure preservation
    assert 'correspondences' in result
    assert 'matching_matrix' in result
    assert 'indices' in result['correspondences']
    assert 'scores' in result['correspondences']
    assert 'valid_mask' in result['correspondences']
    
    # Check shapes
    assert result['correspondences']['indices'].shape == (2, 256, 2)
    assert result['correspondences']['scores'].shape == (256, 2)
    assert result['correspondences']['valid_mask'].shape == (256, 2)
    assert result['matching_matrix'].shape == (100, 100, 2)
    
    # Check data types
    assert result['correspondences']['indices'].dtype == torch.long
    assert result['correspondences']['scores'].dtype == torch.float32
    assert result['correspondences']['valid_mask'].dtype == torch.bool
    assert result['matching_matrix'].dtype == torch.float32


def test_buffer_stack_multi_scale_features():
    """Test buffer_stack with multi-scale feature pyramids."""
    # Multi-scale feature structure
    sample1 = {
        'features': {
            'scale_0': {  # Original resolution
                'positions': torch.randn(2048, 3, dtype=torch.float32),
                'features': torch.randn(2048, 32, dtype=torch.float32)
            },
            'scale_1': {  # 1/2 resolution
                'positions': torch.randn(1024, 3, dtype=torch.float32),
                'features': torch.randn(1024, 64, dtype=torch.float32)
            },
            'scale_2': {  # 1/4 resolution
                'positions': torch.randn(512, 3, dtype=torch.float32),
                'features': torch.randn(512, 128, dtype=torch.float32)
            }
        },
        'downsampling_indices': {
            'scale_0_to_1': torch.randint(0, 2048, (1024,), dtype=torch.long),
            'scale_1_to_2': torch.randint(0, 1024, (512,), dtype=torch.long)
        }
    }
    
    sample2 = {
        'features': {
            'scale_0': {
                'positions': torch.randn(2048, 3, dtype=torch.float32),
                'features': torch.randn(2048, 32, dtype=torch.float32)
            },
            'scale_1': {
                'positions': torch.randn(1024, 3, dtype=torch.float32),
                'features': torch.randn(1024, 64, dtype=torch.float32)
            },
            'scale_2': {
                'positions': torch.randn(512, 3, dtype=torch.float32),
                'features': torch.randn(512, 128, dtype=torch.float32)
            }
        },
        'downsampling_indices': {
            'scale_0_to_1': torch.randint(0, 2048, (1024,), dtype=torch.long),
            'scale_1_to_2': torch.randint(0, 1024, (512,), dtype=torch.long)
        }
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check nested structure
    assert 'features' in result
    assert 'downsampling_indices' in result
    
    # Check multi-scale features
    for scale in ['scale_0', 'scale_1', 'scale_2']:
        assert scale in result['features']
        assert 'positions' in result['features'][scale]
        assert 'features' in result['features'][scale]
    
    # Check shapes are preserved (with stacking dimension)
    assert result['features']['scale_0']['positions'].shape == (2048, 3, 2)
    assert result['features']['scale_0']['features'].shape == (2048, 32, 2)
    assert result['features']['scale_1']['positions'].shape == (1024, 3, 2)
    assert result['features']['scale_1']['features'].shape == (1024, 64, 2)
    assert result['features']['scale_2']['positions'].shape == (512, 3, 2)
    assert result['features']['scale_2']['features'].shape == (512, 128, 2)
    
    # Check downsampling indices
    assert result['downsampling_indices']['scale_0_to_1'].shape == (1024, 2)
    assert result['downsampling_indices']['scale_1_to_2'].shape == (512, 2)


def test_buffer_stack_attention_weights():
    """Test buffer_stack with attention weight structures."""
    # Attention weight structure from transformer layers
    sample1 = {
        'attention': {
            'layer_0': {
                'self_attention': torch.rand(512, 512, dtype=torch.float32),
                'cross_attention': torch.rand(512, 1024, dtype=torch.float32),
                'head_weights': torch.rand(8, 512, 512, dtype=torch.float32)  # Multi-head
            },
            'layer_1': {
                'self_attention': torch.rand(512, 512, dtype=torch.float32),
                'cross_attention': torch.rand(512, 1024, dtype=torch.float32),
                'head_weights': torch.rand(8, 512, 512, dtype=torch.float32)
            }
        },
        'attention_mask': torch.ones(512, dtype=torch.bool)
    }
    
    sample2 = {
        'attention': {
            'layer_0': {
                'self_attention': torch.rand(512, 512, dtype=torch.float32),
                'cross_attention': torch.rand(512, 1024, dtype=torch.float32),
                'head_weights': torch.rand(8, 512, 512, dtype=torch.float32)
            },
            'layer_1': {
                'self_attention': torch.rand(512, 512, dtype=torch.float32),
                'cross_attention': torch.rand(512, 1024, dtype=torch.float32),
                'head_weights': torch.rand(8, 512, 512, dtype=torch.float32)
            }
        },
        'attention_mask': torch.ones(512, dtype=torch.bool)
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check structure preservation
    assert 'attention' in result
    assert 'attention_mask' in result
    
    # Check layer structure
    for layer in ['layer_0', 'layer_1']:
        assert layer in result['attention']
        assert 'self_attention' in result['attention'][layer]
        assert 'cross_attention' in result['attention'][layer]
        assert 'head_weights' in result['attention'][layer]
        
        # Check shapes
        assert result['attention'][layer]['self_attention'].shape == (512, 512, 2)
        assert result['attention'][layer]['cross_attention'].shape == (512, 1024, 2)
        assert result['attention'][layer]['head_weights'].shape == (8, 512, 512, 2)
    
    # Check attention mask
    assert result['attention_mask'].shape == (512, 2)
    assert result['attention_mask'].dtype == torch.bool


def test_buffer_stack_loss_components():
    """Test buffer_stack with loss component structures."""
    # Loss components typically computed during training
    sample1 = {
        'losses': {
            'registration_loss': torch.tensor(0.5, dtype=torch.float32),
            'feature_loss': torch.tensor(0.3, dtype=torch.float32),
            'correspondence_loss': torch.tensor(0.2, dtype=torch.float32)
        },
        'metrics': {
            'rotation_error': torch.tensor(5.2, dtype=torch.float32),  # degrees
            'translation_error': torch.tensor(0.1, dtype=torch.float32),  # meters
            'rmse': torch.tensor(0.05, dtype=torch.float32),
            'inlier_ratio': torch.tensor(0.85, dtype=torch.float32)
        }
    }
    
    sample2 = {
        'losses': {
            'registration_loss': torch.tensor(0.4, dtype=torch.float32),
            'feature_loss': torch.tensor(0.25, dtype=torch.float32),
            'correspondence_loss': torch.tensor(0.18, dtype=torch.float32)
        },
        'metrics': {
            'rotation_error': torch.tensor(4.8, dtype=torch.float32),
            'translation_error': torch.tensor(0.08, dtype=torch.float32),
            'rmse': torch.tensor(0.04, dtype=torch.float32),
            'inlier_ratio': torch.tensor(0.88, dtype=torch.float32)
        }
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check structure preservation
    assert 'losses' in result
    assert 'metrics' in result
    
    # Check loss components become tensors (scalar tensors become stacked tensors)
    assert torch.allclose(result['losses']['registration_loss'], torch.tensor([0.5, 0.4]))
    assert torch.allclose(result['losses']['feature_loss'], torch.tensor([0.3, 0.25]))
    assert torch.allclose(result['losses']['correspondence_loss'], torch.tensor([0.2, 0.18]))
    
    # Check metrics become tensors
    assert torch.allclose(result['metrics']['rotation_error'], torch.tensor([5.2, 4.8]))
    assert torch.allclose(result['metrics']['translation_error'], torch.tensor([0.1, 0.08]))
    assert torch.allclose(result['metrics']['rmse'], torch.tensor([0.05, 0.04]))
    assert torch.allclose(result['metrics']['inlier_ratio'], torch.tensor([0.85, 0.88]))


def test_buffer_stack_metadata():
    """Test buffer_stack with metadata structures."""
    # Metadata typically included with point cloud registration data
    sample1 = {
        'metadata': {
            'scene_id': 'scene_001',
            'frame_ids': [10, 20],  # Source and target frame IDs
            'timestamp': 1634567890.5,
            'num_points': {'src': 2048, 'tgt': 1856},
            'sensor_params': {
                'fov': 90.0,
                'range_max': 80.0,
                'noise_std': 0.01
            },
            'preprocessing': {
                'voxel_size': 0.05,
                'outlier_removal': True,
                'normal_estimation': True
            }
        }
    }
    
    sample2 = {
        'metadata': {
            'scene_id': 'scene_002',
            'frame_ids': [15, 25],
            'timestamp': 1634567895.7,
            'num_points': {'src': 1920, 'tgt': 2100},
            'sensor_params': {
                'fov': 90.0,
                'range_max': 80.0,
                'noise_std': 0.01
            },
            'preprocessing': {
                'voxel_size': 0.05,
                'outlier_removal': True,
                'normal_estimation': True
            }
        }
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check metadata structure
    assert 'metadata' in result
    
    # Check that lists are properly created
    assert result['metadata']['scene_id'] == ['scene_001', 'scene_002']
    assert result['metadata']['frame_ids'] == [[10, 15], [20, 25]]  # Transposed by buffer_stack
    assert result['metadata']['timestamp'] == [1634567890.5, 1634567895.7]
    
    # Check nested dictionaries
    assert result['metadata']['num_points']['src'] == [2048, 1920]
    assert result['metadata']['num_points']['tgt'] == [1856, 2100]
    
    # Check that identical values are correctly stacked
    assert result['metadata']['sensor_params']['fov'] == [90.0, 90.0]
    assert result['metadata']['sensor_params']['range_max'] == [80.0, 80.0]
    assert result['metadata']['sensor_params']['noise_std'] == [0.01, 0.01]
    
    # Check boolean values
    assert result['metadata']['preprocessing']['outlier_removal'] == [True, True]
    assert result['metadata']['preprocessing']['normal_estimation'] == [True, True]


@pytest.mark.parametrize("num_samples", [1, 2, 3, 5, 8])
def test_buffer_stack_variable_batch_sizes(num_samples):
    """Test buffer_stack with different numbers of samples."""
    # Create variable number of samples
    samples = []
    for i in range(num_samples):
        sample = {
            'points': torch.randn(1024, 3, dtype=torch.float32),
            'features': torch.randn(1024, 64, dtype=torch.float32),
            'id': i,
            'valid': i % 2 == 0  # Alternating boolean values
        }
        samples.append(sample)
    
    result = buffer_stack(samples)
    
    # Check structure
    assert 'points' in result
    assert 'features' in result
    assert 'id' in result
    assert 'valid' in result
    
    # Check shapes (with batch dimension from stacking)
    assert result['points'].shape == (1024, 3, num_samples)
    assert result['features'].shape == (1024, 64, num_samples)
    
    # Check that scalar values become lists of correct length
    assert len(result['id']) == num_samples
    assert len(result['valid']) == num_samples
    assert result['id'] == list(range(num_samples))
    
    # Check boolean pattern
    expected_valid = [i % 2 == 0 for i in range(num_samples)]
    assert result['valid'] == expected_valid


def test_buffer_stack_mixed_precision():
    """Test buffer_stack with mixed precision data types."""
    # Different precision types commonly used in point cloud processing
    sample1 = {
        'coordinates': torch.randn(500, 3, dtype=torch.float64),  # High precision coordinates
        'features': torch.randn(500, 32, dtype=torch.float32),   # Standard precision features
        'indices': torch.randint(0, 1000, (500,), dtype=torch.int32),  # 32-bit indices
        'labels': torch.randint(0, 10, (500,), dtype=torch.int8),     # Compact labels
        'confidence': torch.rand(500, dtype=torch.float16)           # Half precision confidence
    }
    
    sample2 = {
        'coordinates': torch.randn(500, 3, dtype=torch.float64),
        'features': torch.randn(500, 32, dtype=torch.float32),
        'indices': torch.randint(0, 1000, (500,), dtype=torch.int32),
        'labels': torch.randint(0, 10, (500,), dtype=torch.int8),
        'confidence': torch.rand(500, dtype=torch.float16)
    }
    
    result = buffer_stack([sample1, sample2])
    
    # Check that data types are preserved
    assert result['coordinates'].dtype == torch.float64
    assert result['features'].dtype == torch.float32
    assert result['indices'].dtype == torch.int32
    assert result['labels'].dtype == torch.int8
    assert result['confidence'].dtype == torch.float16
    
    # Check shapes (with stacking dimension)
    assert result['coordinates'].shape == (500, 3, 2)
    assert result['features'].shape == (500, 32, 2)
    assert result['indices'].shape == (500, 2)
    assert result['labels'].shape == (500, 2)
    assert result['confidence'].shape == (500, 2)
