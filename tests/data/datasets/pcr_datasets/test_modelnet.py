import numpy as np
import pytest
from data.datasets.pcr_datasets.modelnet import ModelNet40Dataset

def test_dataset_structure():
    """Test the structure and content of dataset outputs."""
    # Initialize dataset
    dataset = ModelNet40Dataset(
        root_dir='./data/datasets/soft_links/ModelNet40',
        split='train',
        num_points=1024,
        categories=None,  # Use all categories
        use_normals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        partial_p_keep=0.7
    )
    
    # Test all samples
    for idx in range(len(dataset)):
        data = dataset[idx]
        
        # Check if all required keys are present
        required_keys = [
            'src_points', 'tgt_points', 'transform',
            'category', 'category_id'
        ]
        if dataset.use_normals:
            required_keys.extend(['src_normals', 'tgt_normals'])
            
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in sample {idx}"
            
        # Check data types and shapes
        assert isinstance(data['src_points'], np.ndarray), f"src_points is not np.ndarray in sample {idx}"
        assert isinstance(data['tgt_points'], np.ndarray), f"tgt_points is not np.ndarray in sample {idx}"
        assert isinstance(data['transform'], np.ndarray), f"transform is not np.ndarray in sample {idx}"
        assert isinstance(data['category'], str), f"category is not str in sample {idx}"
        assert isinstance(data['category_id'], int), f"category_id is not int in sample {idx}"
        
        if dataset.use_normals:
            assert isinstance(data['src_normals'], np.ndarray), f"src_normals is not np.ndarray in sample {idx}"
            assert isinstance(data['tgt_normals'], np.ndarray), f"tgt_normals is not np.ndarray in sample {idx}"
        
        # Check array shapes
        assert data['src_points'].shape[1] == 3, f"src_points shape incorrect in sample {idx}"
        assert data['tgt_points'].shape[1] == 3, f"tgt_points shape incorrect in sample {idx}"
        assert data['transform'].shape == (4, 4), f"transform shape incorrect in sample {idx}"
        
        if dataset.use_normals:
            assert data['src_normals'].shape[1] == 3, f"src_normals shape incorrect in sample {idx}"
            assert data['tgt_normals'].shape[1] == 3, f"tgt_normals shape incorrect in sample {idx}"
            
        # Check numeric ranges and properties
        # Category ID should be valid
        assert 0 <= data['category_id'] < len(dataset.ALL_CATEGORIES), f"category_id out of range in sample {idx}"
        assert data['category'] == dataset.ALL_CATEGORIES[data['category_id']], f"category mismatch in sample {idx}"
        
        # Check transformation matrix properties
        transform = data['transform']
        rotation = transform[:3, :3]
        assert np.allclose(np.eye(3), rotation @ rotation.T), f"rotation matrix not orthogonal in sample {idx}"
        assert np.allclose(1.0, np.linalg.det(rotation)), f"rotation matrix determinant not 1 in sample {idx}"
        
        # Check normal vectors if used
        if dataset.use_normals:
            # Normals should be unit vectors
            src_norms = np.linalg.norm(data['src_normals'], axis=1)
            tgt_norms = np.linalg.norm(data['tgt_normals'], axis=1)
            assert np.allclose(1.0, src_norms, atol=1e-6), f"src_normals not unit vectors in sample {idx}"
            assert np.allclose(1.0, tgt_norms, atol=1e-6), f"tgt_normals not unit vectors in sample {idx}"
        
        # Check data types
        assert data['src_points'].dtype == np.float32, f"src_points dtype incorrect in sample {idx}"
        assert data['tgt_points'].dtype == np.float32, f"tgt_points dtype incorrect in sample {idx}"
        assert data['transform'].dtype == np.float32, f"transform dtype incorrect in sample {idx}"
        if dataset.use_normals:
            assert data['src_normals'].dtype == np.float32, f"src_normals dtype incorrect in sample {idx}"
            assert data['tgt_normals'].dtype == np.float32, f"tgt_normals dtype incorrect in sample {idx}"
            
        # Check number of points
        if dataset.num_points is not None:
            assert data['src_points'].shape[0] <= dataset.num_points, f"src_points exceeds num_points in sample {idx}"
            assert data['tgt_points'].shape[0] <= dataset.num_points, f"tgt_points exceeds num_points in sample {idx}"
