import os
import unittest
import numpy as np
import torch
from data.datasets.pcr_datasets.modelnet import ModelNet40Dataset

class TestModelNet40Dataset(unittest.TestCase):
    """Test cases for ModelNet40Dataset."""
    
    def test_dataset_structure(self):
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
        
        # Get a data point
        data = dataset[0]
        
        # Check if all required keys are present
        required_keys = [
            'src_points', 'tgt_points', 'transform',
            'category', 'category_id'
        ]
        if dataset.use_normals:
            required_keys.extend(['src_normals', 'tgt_normals'])
            
        for key in required_keys:
            self.assertIn(key, data)
            
        # Check data types and shapes
        self.assertIsInstance(data['src_points'], np.ndarray)
        self.assertIsInstance(data['tgt_points'], np.ndarray)
        self.assertIsInstance(data['transform'], np.ndarray)
        self.assertIsInstance(data['category'], str)
        self.assertIsInstance(data['category_id'], int)
        
        if dataset.use_normals:
            self.assertIsInstance(data['src_normals'], np.ndarray)
            self.assertIsInstance(data['tgt_normals'], np.ndarray)
        
        # Check array shapes
        self.assertEqual(data['src_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['tgt_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['transform'].shape, (4, 4))  # 4x4
        
        if dataset.use_normals:
            self.assertEqual(data['src_normals'].shape[1], 3)  # Nx3
            self.assertEqual(data['tgt_normals'].shape[1], 3)  # Nx3
            
        # Check numeric ranges and properties
        # Category ID should be valid
        self.assertTrue(0 <= data['category_id'] < len(dataset.ALL_CATEGORIES))
        self.assertEqual(data['category'], dataset.ALL_CATEGORIES[data['category_id']])
        
        # Check transformation matrix properties
        transform = data['transform']
        rotation = transform[:3, :3]
        self.assertTrue(np.allclose(np.eye(3), rotation @ rotation.T))  # Rotation matrix should be orthogonal
        self.assertTrue(np.allclose(1.0, np.linalg.det(rotation)))  # Determinant should be 1
        
        # Check normal vectors if used
        if dataset.use_normals:
            # Normals should be unit vectors
            src_norms = np.linalg.norm(data['src_normals'], axis=1)
            tgt_norms = np.linalg.norm(data['tgt_normals'], axis=1)
            self.assertTrue(np.allclose(1.0, src_norms, atol=1e-6))
            self.assertTrue(np.allclose(1.0, tgt_norms, atol=1e-6))
        
        # Check data types
        self.assertEqual(data['src_points'].dtype, np.float32)
        self.assertEqual(data['tgt_points'].dtype, np.float32)
        self.assertEqual(data['transform'].dtype, np.float32)
        if dataset.use_normals:
            self.assertEqual(data['src_normals'].dtype, np.float32)
            self.assertEqual(data['tgt_normals'].dtype, np.float32)
            
        # Check number of points
        if dataset.num_points is not None:
            self.assertLessEqual(data['src_points'].shape[0], dataset.num_points)
            self.assertLessEqual(data['tgt_points'].shape[0], dataset.num_points)

if __name__ == '__main__':
    unittest.main() 