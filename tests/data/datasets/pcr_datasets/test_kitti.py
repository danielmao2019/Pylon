import os
import unittest
import numpy as np
import torch
from data.datasets.pcr_datasets.kitti import KITTIDataset

class TestKITTIDataset(unittest.TestCase):
    """Test cases for KITTIDataset."""
    
    def test_dataset_structure(self):
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
        
        # Get a data point
        data = dataset[0]
        
        # Check if all required keys are present
        required_keys = [
            'src_points', 'tgt_points', 'transform',
            'sequence', 'src_frame', 'tgt_frame', 'distance'
        ]
        for key in required_keys:
            self.assertIn(key, data)
            
        # Check data types and shapes
        self.assertIsInstance(data['src_points'], np.ndarray)
        self.assertIsInstance(data['tgt_points'], np.ndarray)
        self.assertIsInstance(data['transform'], np.ndarray)
        self.assertIsInstance(data['sequence'], str)
        self.assertIsInstance(data['src_frame'], str)
        self.assertIsInstance(data['tgt_frame'], str)
        self.assertIsInstance(data['distance'], float)
        
        # Check array shapes
        self.assertEqual(data['src_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['tgt_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['transform'].shape, (4, 4))  # 4x4
        
        # Check numeric ranges and properties
        self.assertTrue(data['distance'] >= 0.0)  # Distance should be non-negative
        self.assertTrue(data['distance'] <= dataset.max_dist)  # Distance should be within max_dist
        
        # Check transformation matrix properties
        transform = data['transform']
        rotation = transform[:3, :3]
        self.assertTrue(np.allclose(np.eye(3), rotation @ rotation.T))  # Rotation matrix should be orthogonal
        self.assertTrue(np.allclose(1.0, np.linalg.det(rotation)))  # Determinant should be 1
        
        # Check sequence format
        self.assertTrue(data['sequence'] in dataset.TRAIN_SEQUENCES)
        
        # Check frame names format
        self.assertTrue(data['src_frame'].endswith('.bin'))
        self.assertTrue(data['tgt_frame'].endswith('.bin'))
        
        # Check data types
        self.assertEqual(data['src_points'].dtype, np.float32)
        self.assertEqual(data['tgt_points'].dtype, np.float32)
        self.assertEqual(data['transform'].dtype, np.float32)

if __name__ == '__main__':
    unittest.main() 