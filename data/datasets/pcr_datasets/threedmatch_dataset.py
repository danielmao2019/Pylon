import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class ThreeDMatchDataset(Dataset):
    """3DMatch dataset for point cloud registration.
    
    This dataset contains RGB-D scans of real-world indoor scenes from the 3DMatch benchmark.
    It is commonly used for evaluating point cloud registration algorithms.
    """
    
    def __init__(
        self,
        root_dir,
        split='train',
        num_points=5000,
        use_mutuals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        overlap_threshold=0.3
    ):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Path to dataset root directory
            split (str): Dataset split ('train', 'val', 'test')
            num_points (int): Number of points to sample from each point cloud
            use_mutuals (bool): Whether to use mutual nearest neighbors for correspondences
            augment (bool): Whether to apply data augmentation
            rot_mag (float): Maximum rotation angle for augmentation (degrees)
            trans_mag (float): Maximum translation magnitude for augmentation
            noise_std (float): Standard deviation of Gaussian noise for augmentation
            overlap_threshold (float): Minimum overlap ratio between point cloud pairs
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.use_mutuals = use_mutuals
        self.augment = augment
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.overlap_threshold = overlap_threshold
        
        # Load metadata
        self.metadata_dir = os.path.join(root_dir, 'metadata')
        self.data_dir = os.path.join(root_dir, 'data')
        
        with open(os.path.join(self.metadata_dir, f'{split}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
                
    def __len__(self):
        return len(self.metadata_list)
    
    def _load_point_cloud(self, file_name):
        """Load point cloud and sample points if needed."""
        points = torch.load(os.path.join(self.data_dir, file_name))
        if self.num_points is not None and points.shape[0] > self.num_points:
            indices = np.random.permutation(points.shape[0])[:self.num_points]
            points = points[indices]
        return points
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        """Apply data augmentation to point clouds."""
        # Random rotation
        if self.augment:
            aug_angle = np.random.uniform(-self.rot_mag, self.rot_mag)
            aug_axis = np.random.randn(3)
            aug_axis = aug_axis / np.linalg.norm(aug_axis)
            aug_rotation = self._get_rotation_matrix(aug_axis, aug_angle)
            
            # Apply to either source or reference randomly
            if np.random.random() > 0.5:
                ref_points = np.matmul(ref_points, aug_rotation.T)
                rotation = np.matmul(aug_rotation, rotation)
                translation = np.matmul(aug_rotation, translation)
            else:
                src_points = np.matmul(src_points, aug_rotation.T)
                rotation = np.matmul(rotation, aug_rotation.T)
            
            # Add random noise
            ref_points += np.random.normal(0, self.noise_std, ref_points.shape)
            src_points += np.random.normal(0, self.noise_std, src_points.shape)
            
        return ref_points, src_points, rotation, translation
    
    def _get_rotation_matrix(self, axis, angle):
        """Get rotation matrix from axis and angle."""
        angle = np.deg2rad(angle)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)
        return rotation
    
    def __getitem__(self, index):
        """Get a pair of point clouds and their relative transformation."""
        metadata = self.metadata_list[index]
        
        # Load point clouds
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])
        
        # Get ground truth transformation
        rotation = metadata['rotation']
        translation = metadata['translation']
        
        # Apply augmentation
        if self.augment:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
        
        # Prepare output dictionary
        data_dict = {
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'rotation': rotation.astype(np.float32),
            'translation': translation.astype(np.float32),
            'scene_name': metadata['scene_name'],
            'ref_frame': metadata['frag_id0'],
            'src_frame': metadata['frag_id1'],
            'overlap': metadata['overlap']
        }
        
        return data_dict 