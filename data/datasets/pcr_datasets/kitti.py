import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

class KITTIDataset(Dataset):
    """KITTI dataset for point cloud registration.
    
    This dataset contains LiDAR scans from the KITTI odometry benchmark.
    It is commonly used for evaluating point cloud registration algorithms
    in autonomous driving scenarios.
    """
    
    TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05']
    VAL_SEQUENCES = ['06', '07']
    TEST_SEQUENCES = ['08', '09', '10']
    
    def __init__(
        self,
        root_dir,
        split='train',
        num_points=None,
        use_mutuals=True,
        augment=True,
        rot_mag=45.0,
        trans_mag=0.5,
        noise_std=0.01,
        min_overlap=0.3,
        max_dist=5.0
    ):
        """Initialize the dataset.
        
        Args:
            root_dir (str): Path to dataset root directory
            split (str): Dataset split ('train', 'val', 'test')
            num_points (int): Number of points to sample (None for no sampling)
            use_mutuals (bool): Whether to use mutual nearest neighbors
            augment (bool): Whether to apply data augmentation
            rot_mag (float): Maximum rotation angle for augmentation (degrees)
            trans_mag (float): Maximum translation magnitude for augmentation
            noise_std (float): Standard deviation of Gaussian noise
            min_overlap (float): Minimum overlap ratio between point cloud pairs
            max_dist (float): Maximum distance between frames to be considered as pairs
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.use_mutuals = use_mutuals
        self.augment = augment
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.noise_std = noise_std
        self.min_overlap = min_overlap
        self.max_dist = max_dist
        
        # Get sequences for the split
        if split == 'train':
            self.sequences = self.TRAIN_SEQUENCES
        elif split == 'val':
            self.sequences = self.VAL_SEQUENCES
        else:
            self.sequences = self.TEST_SEQUENCES
            
        # Load point cloud pairs
        self.pairs = []
        for sequence in self.sequences:
            seq_dir = os.path.join(root_dir, 'sequences', sequence)
            calib_file = os.path.join(seq_dir, 'calib.txt')
            poses_file = os.path.join(root_dir, 'poses', f'{sequence}.txt')
            
            # Load calibration
            with open(calib_file, 'r') as f:
                lines = f.readlines()
                Tr = np.array([float(x) for x in lines[4].strip().split()[1:]]).reshape(3, 4)
                Tr = np.vstack((Tr, [0, 0, 0, 1]))
            
            # Load poses
            poses = np.loadtxt(poses_file)
            
            # Get valid pairs
            velodyne_dir = os.path.join(seq_dir, 'velodyne')
            scans = sorted(os.listdir(velodyne_dir))
            
            for i in range(len(scans)):
                for j in range(i + 1, len(scans)):
                    # Check if frames are close enough
                    dist = np.linalg.norm(poses[j][:3, 3] - poses[i][:3, 3])
                    if dist > self.max_dist:
                        continue
                        
                    # Compute relative pose
                    pose_i = np.vstack((poses[i], [0, 0, 0, 1]))
                    pose_j = np.vstack((poses[j], [0, 0, 0, 1]))
                    relative_pose = np.linalg.inv(pose_i) @ pose_j
                    
                    self.pairs.append({
                        'sequence': sequence,
                        'src_frame': scans[i],
                        'tgt_frame': scans[j],
                        'transform': relative_pose,
                        'dist': dist
                    })
                    
    def __len__(self):
        return len(self.pairs)
    
    def _load_point_cloud(self, sequence, frame):
        """Load point cloud from binary file."""
        file_path = os.path.join(self.root_dir, 'sequences', sequence, 'velodyne', frame)
        scan = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]
        
        if self.num_points is not None and points.shape[0] > self.num_points:
            indices = np.random.permutation(points.shape[0])[:self.num_points]
            points = points[indices]
            
        return points
    
    def _augment_point_cloud(self, ref_points, src_points, transform):
        """Apply data augmentation to point clouds."""
        if not self.augment:
            return ref_points, src_points, transform
            
        # Extract rotation and translation
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        
        # Random rotation
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
        
        # Update transform
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        return ref_points, src_points, transform
    
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
        pair = self.pairs[index]
        
        # Load point clouds
        src_points = self._load_point_cloud(pair['sequence'], pair['src_frame'])
        tgt_points = self._load_point_cloud(pair['sequence'], pair['tgt_frame'])
        
        # Apply augmentation
        src_points, tgt_points, transform = self._augment_point_cloud(
            src_points, tgt_points, pair['transform']
        )
        
        # Prepare output dictionary
        data_dict = {
            'src_points': src_points.astype(np.float32),
            'tgt_points': tgt_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'sequence': pair['sequence'],
            'src_frame': pair['src_frame'],
            'tgt_frame': pair['tgt_frame'],
            'distance': pair['dist']
        }
        
        return data_dict 