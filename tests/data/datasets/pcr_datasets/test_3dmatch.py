import numpy as np
import pytest
import os
from data.datasets.pcr_datasets.threedmatch import ThreeDMatchDataset

def test_dataset_structure():
    """Test the structure and content of dataset outputs."""
    # Initialize dataset
    dataset = ThreeDMatchDataset(
        root_dir='./data/datasets/soft_links/3dmatch',
        split='train',
        num_points=5000,
        use_mutuals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        overlap_threshold=0.3
    )
    
    # Test all samples
    for idx in range(len(dataset)):
        data = dataset[idx]
        
        # Check if all required keys are present
        required_keys = [
            'ref_points', 'src_points', 'rotation', 'translation',
            'scene_name', 'ref_frame', 'src_frame'
        ]
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in sample {idx}"
            
        # Check data types and shapes
        assert isinstance(data['ref_points'], np.ndarray), f"ref_points is not np.ndarray in sample {idx}"
        assert isinstance(data['src_points'], np.ndarray), f"src_points is not np.ndarray in sample {idx}"
        assert isinstance(data['rotation'], np.ndarray), f"rotation is not np.ndarray in sample {idx}"
        assert isinstance(data['translation'], np.ndarray), f"translation is not np.ndarray in sample {idx}"
        assert isinstance(data['scene_name'], str), f"scene_name is not str in sample {idx}"
        assert isinstance(data['ref_frame'], str), f"ref_frame is not str in sample {idx}"
        assert isinstance(data['src_frame'], str), f"src_frame is not str in sample {idx}"
        
        # Check array shapes
        assert data['ref_points'].shape[1] == 3, f"ref_points shape incorrect in sample {idx}"
        assert data['src_points'].shape[1] == 3, f"src_points shape incorrect in sample {idx}"
        assert data['rotation'].shape == (3, 3), f"rotation shape incorrect in sample {idx}"
        assert data['translation'].shape == (3,), f"translation shape incorrect in sample {idx}"
        
        # Check numeric ranges
        assert np.allclose(np.eye(3), data['rotation'] @ data['rotation'].T), f"rotation matrix not orthogonal in sample {idx}"
        
        # Check data types
        assert data['ref_points'].dtype == np.float32, f"ref_points dtype incorrect in sample {idx}"
        assert data['src_points'].dtype == np.float32, f"src_points dtype incorrect in sample {idx}"
        assert data['rotation'].dtype == np.float32, f"rotation dtype incorrect in sample {idx}"
        assert data['translation'].dtype == np.float32, f"translation dtype incorrect in sample {idx}"
        
        # Check file existence
        scene_dir = os.path.join(dataset.root_dir, data['scene_name'])
        ref_file = os.path.join(scene_dir, data['ref_frame'])
        src_file = os.path.join(scene_dir, data['src_frame'])
        assert os.path.exists(ref_file), f"Reference file {ref_file} does not exist"
        assert os.path.exists(src_file), f"Source file {src_file} does not exist"
