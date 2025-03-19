import os
import unittest
import numpy as np
import torch
from data.datasets.pcr_datasets.threedmatch import ThreeDMatchDataset

class TestThreeDMatchDataset(unittest.TestCase):
    """Test cases for ThreeDMatchDataset."""
    
    def test_dataset_structure(self):
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
        
        # Get a data point
        data = dataset[0]
        
        # Check if all required keys are present
        required_keys = [
            'ref_points', 'src_points', 'rotation', 'translation',
            'scene_name', 'ref_frame', 'src_frame', 'overlap'
        ]
        for key in required_keys:
            self.assertIn(key, data)
            
        # Check data types and shapes
        self.assertIsInstance(data['ref_points'], np.ndarray)
        self.assertIsInstance(data['src_points'], np.ndarray)
        self.assertIsInstance(data['rotation'], np.ndarray)
        self.assertIsInstance(data['translation'], np.ndarray)
        self.assertIsInstance(data['scene_name'], str)
        self.assertIsInstance(data['ref_frame'], str)
        self.assertIsInstance(data['src_frame'], str)
        self.assertIsInstance(data['overlap'], float)
        
        # Check array shapes
        self.assertEqual(data['ref_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['src_points'].shape[1], 3)  # Nx3
        self.assertEqual(data['rotation'].shape, (3, 3))  # 3x3
        self.assertEqual(data['translation'].shape, (3,))  # 3
        
        # Check numeric ranges
        self.assertTrue(0.0 <= data['overlap'] <= 1.0)  # Overlap should be between 0 and 1
        self.assertTrue(np.allclose(np.eye(3), data['rotation'] @ data['rotation'].T))  # Rotation matrix should be orthogonal
        
        # Check data types
        self.assertEqual(data['ref_points'].dtype, np.float32)
        self.assertEqual(data['src_points'].dtype, np.float32)
        self.assertEqual(data['rotation'].dtype, np.float32)
        self.assertEqual(data['translation'].dtype, np.float32)

if __name__ == '__main__':
    unittest.main() 