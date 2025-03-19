import numpy as np
import pytest
from data.datasets.pcr_datasets.kitti import KITTIDataset

def test_dataset_structure():
    """Test the structure and content of dataset outputs."""
    # Initialize dataset
    dataset = KITTIDataset(
        root_dir='./data/datasets/soft_links/KITTI',
        split='train',
        num_points=None,
        use_mutuals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        min_overlap=0.3,
        max_dist=5.0
    )
    
    # Test all samples
    for idx in range(len(dataset)):
        data = dataset[idx]
        
        # Check if all required keys are present
        required_keys = [
            'src_points', 'tgt_points', 'transform',
            'sequence', 'src_frame', 'tgt_frame', 'distance'
        ]
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in sample {idx}"
            
        # Check data types and shapes
        assert isinstance(data['src_points'], np.ndarray), f"src_points is not np.ndarray in sample {idx}"
        assert isinstance(data['tgt_points'], np.ndarray), f"tgt_points is not np.ndarray in sample {idx}"
        assert isinstance(data['transform'], np.ndarray), f"transform is not np.ndarray in sample {idx}"
        assert isinstance(data['sequence'], str), f"sequence is not str in sample {idx}"
        assert isinstance(data['src_frame'], str), f"src_frame is not str in sample {idx}"
        assert isinstance(data['tgt_frame'], str), f"tgt_frame is not str in sample {idx}"
        assert isinstance(data['distance'], float), f"distance is not float in sample {idx}"
        
        # Check array shapes
        assert data['src_points'].shape[1] == 3, f"src_points shape incorrect in sample {idx}"
        assert data['tgt_points'].shape[1] == 3, f"tgt_points shape incorrect in sample {idx}"
        assert data['transform'].shape == (4, 4), f"transform shape incorrect in sample {idx}"
        
        # Check numeric ranges and properties
        assert data['distance'] >= 0.0, f"distance negative in sample {idx}"
        assert data['distance'] <= dataset.max_dist, f"distance exceeds max_dist in sample {idx}"
        
        # Check transformation matrix properties
        transform = data['transform']
        rotation = transform[:3, :3]
        assert np.allclose(np.eye(3), rotation @ rotation.T), f"rotation matrix not orthogonal in sample {idx}"
        assert np.allclose(1.0, np.linalg.det(rotation)), f"rotation matrix determinant not 1 in sample {idx}"
        
        # Check sequence format
        assert data['sequence'] in dataset.TRAIN_SEQUENCES, f"sequence not in TRAIN_SEQUENCES in sample {idx}"
        
        # Check frame names format
        assert data['src_frame'].endswith('.bin'), f"src_frame not .bin in sample {idx}"
        assert data['tgt_frame'].endswith('.bin'), f"tgt_frame not .bin in sample {idx}"
        
        # Check data types
        assert data['src_points'].dtype == np.float32, f"src_points dtype incorrect in sample {idx}"
        assert data['tgt_points'].dtype == np.float32, f"tgt_points dtype incorrect in sample {idx}"
        assert data['transform'].dtype == np.float32, f"transform dtype incorrect in sample {idx}"
