"""Test datapoint structure and content validation for LiDAR camera pose PCR dataset."""
import random
import torch
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from data.datasets.pcr_datasets.lidar_camera_pose_pcr_dataset import LiDARCameraPosePCRDataset


def validate_inputs(inputs: Dict[str, Any]) -> None:
    """Validate inputs dictionary structure and tensor properties."""
    # Check required keys
    required_keys = {'src_pc', 'tgt_pc', 'correspondences'}
    assert inputs.keys() == required_keys, f"Expected keys {required_keys}, got {inputs.keys()}"
    
    # Validate source point cloud
    assert isinstance(inputs['src_pc'], dict), f"src_pc must be dict, got {type(inputs['src_pc'])}"
    assert 'pos' in inputs['src_pc'], "src_pc must have 'pos' key"
    assert isinstance(inputs['src_pc']['pos'], torch.Tensor), f"src_pc['pos'] must be tensor, got {type(inputs['src_pc']['pos'])}"
    assert inputs['src_pc']['pos'].shape[1] == 3, f"src_pc['pos'] must have shape (N, 3), got {inputs['src_pc']['pos'].shape}"
    assert inputs['src_pc']['pos'].dtype == torch.float32, f"src_pc['pos'] must be float32, got {inputs['src_pc']['pos'].dtype}"
    assert inputs['src_pc']['pos'].numel() > 0, "src_pc['pos'] must not be empty"
    
    # Validate target point cloud
    assert isinstance(inputs['tgt_pc'], dict), f"tgt_pc must be dict, got {type(inputs['tgt_pc'])}"
    assert 'pos' in inputs['tgt_pc'], "tgt_pc must have 'pos' key"
    assert isinstance(inputs['tgt_pc']['pos'], torch.Tensor), f"tgt_pc['pos'] must be tensor, got {type(inputs['tgt_pc']['pos'])}"
    assert inputs['tgt_pc']['pos'].shape[1] == 3, f"tgt_pc['pos'] must have shape (N, 3), got {inputs['tgt_pc']['pos'].shape}"
    assert inputs['tgt_pc']['pos'].dtype == torch.float32, f"tgt_pc['pos'] must be float32, got {inputs['tgt_pc']['pos'].dtype}"
    assert inputs['tgt_pc']['pos'].numel() > 0, "tgt_pc['pos'] must not be empty"
    
    # Validate correspondences
    assert isinstance(inputs['correspondences'], torch.Tensor), f"correspondences must be tensor, got {type(inputs['correspondences'])}"
    assert inputs['correspondences'].shape[1] == 2, f"correspondences must have shape (M, 2), got {inputs['correspondences'].shape}"
    assert inputs['correspondences'].dtype == torch.int64, f"correspondences must be int64, got {inputs['correspondences'].dtype}"
    
    # Validate correspondence indices are within bounds
    if inputs['correspondences'].numel() > 0:
        src_indices = inputs['correspondences'][:, 0]
        tgt_indices = inputs['correspondences'][:, 1]
        src_max_idx = inputs['src_pc']['pos'].shape[0] - 1
        tgt_max_idx = inputs['tgt_pc']['pos'].shape[0] - 1
        
        assert src_indices.min() >= 0, f"Source correspondence indices must be >= 0, got min {src_indices.min()}"
        assert src_indices.max() <= src_max_idx, f"Source correspondence indices must be <= {src_max_idx}, got max {src_indices.max()}"
        assert tgt_indices.min() >= 0, f"Target correspondence indices must be >= 0, got min {tgt_indices.min()}"
        assert tgt_indices.max() <= tgt_max_idx, f"Target correspondence indices must be <= {tgt_max_idx}, got max {tgt_indices.max()}"


def validate_labels(labels: Dict[str, Any]) -> None:
    """Validate labels dictionary structure and tensor properties."""
    # Check required keys
    required_keys = {'transform'}
    assert labels.keys() == required_keys, f"Expected keys {required_keys}, got {labels.keys()}"
    
    # Validate transform matrix
    assert isinstance(labels['transform'], torch.Tensor), f"transform must be tensor, got {type(labels['transform'])}"
    assert labels['transform'].shape == (4, 4), f"transform must have shape (4, 4), got {labels['transform'].shape}"
    assert labels['transform'].dtype == torch.float32, f"transform must be float32, got {labels['transform'].dtype}"
    
    # Validate transform matrix properties (should be valid SE(3) transformation)
    transform = labels['transform']
    
    # Check bottom row is [0, 0, 0, 1]
    expected_bottom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
    bottom_row = transform[3, :]
    assert torch.allclose(bottom_row, expected_bottom_row, atol=1e-6), f"Bottom row should be [0, 0, 0, 1], got {bottom_row}"
    
    # Check rotation matrix properties (orthogonal, determinant = 1)
    rotation = transform[:3, :3]
    should_be_identity = torch.mm(rotation, rotation.T)
    identity = torch.eye(3, dtype=torch.float32)
    assert torch.allclose(should_be_identity, identity, atol=1e-5), "Rotation matrix should be orthogonal"
    
    det = torch.det(rotation)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), f"Rotation matrix determinant should be 1, got {det}"


def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    """Validate meta_info dictionary structure and content."""
    # Check required keys (BaseDataset adds 'idx')
    required_keys = {'idx', 'transform_params'}
    assert isinstance(meta_info, dict), f"meta_info must be dict, got {type(meta_info)}"
    assert required_keys.issubset(meta_info.keys()), f"meta_info missing required keys. Expected {required_keys}, got {meta_info.keys()}"
    
    # Validate idx matches datapoint index
    assert meta_info['idx'] == datapoint_idx, f"meta_info['idx'] must match datapoint index {datapoint_idx}, got {meta_info['idx']}"
    
    # Validate transform_params
    transform_params = meta_info['transform_params']
    assert isinstance(transform_params, dict), f"transform_params must be dict, got {type(transform_params)}"
    
    # Check required transform config keys (crop_method removed since it's always LiDAR)
    required_config_keys = {
        'rotation_angles', 'translation',
        'camera_pose_idx', 'sensor_position', 'sensor_euler_angles',
        'lidar_max_range', 'lidar_horizontal_fov', 'lidar_vertical_fov',
        'seed',
    }
    assert required_config_keys.issubset(transform_params.keys()), f"transform_params missing keys: {required_config_keys - set(transform_params.keys())}"
    
    # Validate specific fields
    assert isinstance(transform_params['camera_pose_idx'], int), f"camera_pose_idx must be int, got {type(transform_params['camera_pose_idx'])}"
    assert transform_params['camera_pose_idx'] >= 0, f"camera_pose_idx must be >= 0, got {transform_params['camera_pose_idx']}"
    
    # Validate numeric values
    assert isinstance(transform_params['rotation_angles'], list) and len(transform_params['rotation_angles']) == 3
    assert isinstance(transform_params['translation'], list) and len(transform_params['translation']) == 3
    assert isinstance(transform_params['sensor_position'], list) and len(transform_params['sensor_position']) == 3
    assert isinstance(transform_params['sensor_euler_angles'], list) and len(transform_params['sensor_euler_angles']) == 3


def test_lidar_camera_pose_pcr_dataset_structure(dataset, max_samples, get_samples_to_test):
    """Test the structure and content of LiDAR camera pose PCR dataset outputs."""
    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) > 0, "Dataset should not be empty"
    
    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict), f"Datapoint must be dict, got {type(datapoint)}"
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}, f"Expected keys {{inputs, labels, meta_info}}, got {datapoint.keys()}"
        
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)
    
    # Use command line --samples if provided
    num_samples = get_samples_to_test(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Use threading for performance
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)