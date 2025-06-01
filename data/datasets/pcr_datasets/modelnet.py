import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class ModelNet40Dataset(Dataset):
    """ModelNet40 dataset for point cloud registration.
    
    This dataset contains CAD models from 40 different object categories.
    It is commonly used for evaluating point cloud registration algorithms
    on synthetic data.
    """
    
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox'
    ]
    
    def __init__(
        self,
        root_dir,
        split='train',
        num_points=1024,
        categories=None,
        use_normals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        partial_p_keep=0.7
    ):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Path to dataset root directory
            split (str): Dataset split ('train', 'test')
            num_points (int): Number of points to sample from each point cloud
            categories (list): List of categories to use (None for all)
            use_normals (bool): Whether to include normal vectors
            augment (bool): Whether to apply data augmentation
            rot_mag (float): Maximum rotation angle for augmentation (degrees)
            trans_mag (float): Maximum translation magnitude for augmentation
            noise_std (float): Standard deviation of Gaussian noise
            partial_p_keep (float): Proportion of points to keep during cropping
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.partial_p_keep = partial_p_keep
        
        # Get categories
        if categories is not None:
            assert all(c in self.ALL_CATEGORIES for c in categories)
            self.categories = categories
            self.category_ids = [self.ALL_CATEGORIES.index(c) for c in categories]
        else:
            self.categories = self.ALL_CATEGORIES
            self.category_ids = list(range(len(self.ALL_CATEGORIES)))
            
        # Load data
        self.data = []
        self.labels = []
        
        h5_files = self._get_h5_files()
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                points = f['data'][:]
                if self.use_normals:
                    normals = f['normal'][:]
                    points = np.concatenate([points, normals], axis=-1)
                labels = f['label'][:].flatten()
                
                # Filter by category
                mask = np.isin(labels, self.category_ids)
                points = points[mask]
                labels = labels[mask]
                
                self.data.append(points)
                self.labels.append(labels)
                
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
    def _get_h5_files(self):
        """Get list of HDF5 files for the split."""
        with open(os.path.join(self.root_dir, f'{self.split}_files.txt'), 'r') as f:
            h5_files = [line.strip() for line in f]
            h5_files = [os.path.join(self.root_dir, f) for f in h5_files]
        return h5_files
    
    def __len__(self):
        return len(self.data)
    
    def _sample_points(self, points):
        """Randomly sample points."""
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        return points
    
    def _random_crop(self, points):
        """Randomly crop point cloud."""
        if not self.augment:
            return points
            
        # Get random viewpoint
        viewpoint = np.random.randn(3)
        viewpoint = viewpoint / np.linalg.norm(viewpoint)
        
        # Project points onto viewpoint direction
        proj = points[:, :3] @ viewpoint
        
        # Sort and crop
        indices = np.argsort(proj)
        num_keep = int(len(indices) * self.partial_p_keep)
        keep_indices = indices[:num_keep]
        
        return points[keep_indices]
    
    def _augment_point_cloud(self, points):
        """Apply data augmentation to point cloud."""
        if not self.augment:
            return points
            
        # Add random noise
        points[:, :3] += np.random.normal(0, self.noise_std, points[:, :3].shape)
        
        # Random rotation
        angle = np.random.uniform(-self.rot_mag, self.rot_mag)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        rotation = self._get_rotation_matrix(axis, angle)
        
        # Apply rotation
        points[:, :3] = points[:, :3] @ rotation.T
        if self.use_normals:
            points[:, 3:6] = points[:, 3:6] @ rotation.T
            
        # Random translation
        translation = np.random.uniform(-self.trans_mag, self.trans_mag, 3)
        points[:, :3] += translation
        
        return points
    
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
        """Get a pair of point clouds from the same shape with relative transformation."""
        # Get source point cloud
        src_points = self.data[index].copy()
        label = self.labels[index]
        
        # Sample points
        src_points = self._sample_points(src_points)
        
        # Create target by applying random transform to source
        tgt_points = src_points.copy()
        
        # Random crop both point clouds
        src_points = self._random_crop(src_points)
        tgt_points = self._random_crop(tgt_points)
        
        # Apply augmentation to source
        src_points = self._augment_point_cloud(src_points)
        
        # Get ground truth transformation (from augmentation)
        transform = np.eye(4)
        transform[:3, :3] = self._get_rotation_matrix(
            np.random.randn(3),
            np.random.uniform(-self.rot_mag, self.rot_mag)
        )
        transform[:3, 3] = np.random.uniform(-self.trans_mag, self.trans_mag, 3)
        
        # Apply transform to target
        tgt_points[:, :3] = (transform[:3, :3] @ tgt_points[:, :3].T).T + transform[:3, 3]
        if self.use_normals:
            tgt_points[:, 3:6] = (transform[:3, :3] @ tgt_points[:, 3:6].T).T
        
        # Prepare output dictionary
        data_dict = {
            'src_points': src_points[:, :3].astype(np.float32),
            'tgt_points': tgt_points[:, :3].astype(np.float32),
            'transform': transform.astype(np.float32),
            'category': self.categories[label],
            'category_id': int(label)
        }
        
        if self.use_normals:
            data_dict.update({
                'src_normals': src_points[:, 3:6].astype(np.float32),
                'tgt_normals': tgt_points[:, 3:6].astype(np.float32)
            })
            
        return data_dict 