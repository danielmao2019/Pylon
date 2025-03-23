import os
from typing import Optional
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import KDTree
from data.datasets.base_dataset import BaseDataset
from utils.torch_points3d import GridSampling3D, CylinderSampling


class SynthPCRDataset(BaseDataset):
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {'train': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self, 
        rot_mag: float = 45.0,
        trans_mag: float = 0.5,
        radius: float = 50.0,
        sampling_mode: str = 'random',
        **kwargs,
    ) -> None:
        assert sampling_mode in ['fixed', 'random', 'grid'], \
            f"sampling_mode must be one of ['fixed', 'random', 'grid'], got {sampling_mode}"
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self._radius = radius
        self._sampling_mode = sampling_mode
        self._grid_sampling = GridSampling3D(size=radius/10.0)
        super(SynthPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self):
        """Initialize dataset annotations and prepare centers for sampling."""
        # Get file paths
        self.file_paths = []
        for file in os.listdir(self.data_root):
            if file.endswith('.ply'):
                self.file_paths.append(os.path.join(self.data_root, file))
        
        # Split into train/test if needed
        if self.split in ['train', 'test']:
            np.random.seed(42)
            indices = np.random.permutation(len(self.file_paths))
            split_idx = int(0.8 * len(self.file_paths))
            if self.split == 'train':
                self.file_paths = [self.file_paths[i] for i in indices[:split_idx]]
            else:
                self.file_paths = [self.file_paths[i] for i in indices[split_idx:]]

        # Prepare all potential centers by grid sampling
        all_centers = self._prepare_all_centers()

        # Prepare final annotations based on sampling mode
        if self._sampling_mode == 'fixed':
            self.annotations = self._prepare_fixed_centers(all_centers)
        elif self._sampling_mode == 'random':
            self.annotations = self._prepare_random_centers(all_centers)
        else:  # grid mode
            self.annotations = self._prepare_grid_centers(all_centers)

    def _prepare_all_centers(self):
        """Prepare all potential centers by grid sampling each point cloud."""
        centers_list = []
        for idx, filepath in enumerate(self.file_paths):
            # Load point cloud
            pcd = o3d.io.read_point_cloud(filepath)
            points = torch.from_numpy(np.asarray(pcd.points).astype(np.float32))
            
            # Normalize points
            mean = points.mean(0, keepdim=True)
            points = points - mean

            # Grid sample to get centers
            data_dict = {'pos': points}
            sampled_data = self._grid_sampling(data_dict)
            
            centers = {
                'pos': sampled_data['pos'],
                'idx': idx * torch.ones(len(sampled_data['pos']), dtype=torch.long),
                'filepath': filepath
            }
            centers_list.append(centers)

        # Convert to single dictionary with concatenated tensors
        return {
            'pos': torch.cat([c['pos'] for c in centers_list], dim=0),
            'idx': torch.cat([c['idx'] for c in centers_list], dim=0),
            'filepath': [c['filepath'] for c in centers_list]
        }

    def _prepare_fixed_centers(self, all_centers):
        """Prepare fixed centers using all grid-sampled centers."""
        fixed_centers = []
        for i in range(len(all_centers['pos'])):
            fixed_centers.append({
                'pos': all_centers['pos'][i],
                'idx': all_centers['idx'][i],
                'filepath': all_centers['filepath'][all_centers['idx'][i]]
            })
        return fixed_centers

    def _prepare_random_centers(self, all_centers):
        """Use all grid-sampled centers but will sample randomly during loading."""
        return self._prepare_fixed_centers(all_centers)

    def _prepare_grid_centers(self, all_centers):
        """Use all grid-sampled centers."""
        return self._prepare_fixed_centers(all_centers)

    def _load_datapoint(self, idx):
        """Load a datapoint using sampling center and generate synthetic pair.
        
        Returns:
            inputs: Dict containing source and target point clouds
            labels: Dict containing transformation matrix
            meta_info: Dict containing additional information
        """
        # Get annotation for this index
        annotation = self.annotations[idx]
        center = annotation['pos']
        
        # Load and sample source point cloud
        src_data = self._load_and_sample_pointcloud(annotation['filepath'], center)
        
        # Generate random transformation
        rot = np.random.uniform(-self.rot_mag, self.rot_mag, 3)
        trans = np.random.uniform(-self.trans_mag, self.trans_mag, 3)
        
        # Create rotation matrix (using Euler angles)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(np.radians(rot[0])), -np.sin(np.radians(rot[0]))],
                      [0, np.sin(np.radians(rot[0])), np.cos(np.radians(rot[0]))]])
        Ry = np.array([[np.cos(np.radians(rot[1])), 0, np.sin(np.radians(rot[1]))],
                      [0, 1, 0],
                      [-np.sin(np.radians(rot[1])), 0, np.cos(np.radians(rot[1]))]])
        Rz = np.array([[np.cos(np.radians(rot[2])), -np.sin(np.radians(rot[2])), 0],
                      [np.sin(np.radians(rot[2])), np.cos(np.radians(rot[2])), 0],
                      [0, 0, 1]])
        R = Rx @ Ry @ Rz

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = trans

        # Apply transformation to create target point cloud
        src_points = src_data['pos']
        tgt_points = (R @ src_points.T).T + trans

        # Convert to torch tensors with features
        src_features = torch.ones((src_points.shape[0], 1), dtype=torch.float32)
        tgt_features = torch.ones((tgt_points.shape[0], 1), dtype=torch.float32)
        transform = torch.from_numpy(transform.astype(np.float32))

        inputs = {
            'src_pc': {
                'pos': torch.from_numpy(src_points.astype(np.float32)),
                'feat': src_features
            },
            'tgt_pc': {
                'pos': torch.from_numpy(tgt_points.astype(np.float32)),
                'feat': tgt_features
            }
        }
        
        labels = {
            'transform': torch.from_numpy(transform.astype(np.float32))  # Remove batch dimension
        }
        
        meta_info = {
            'idx': idx,
            'center_pos': center,
            'point_idx': src_data['point_idx'],
            'filepath': annotation['filepath']
        }

        return inputs, labels, meta_info

    def _load_and_sample_pointcloud(self, filepath, center):
        """Load point cloud and sample points within cylinder.
        
        Args:
            filepath: Path to point cloud file
            center: Center position for cylinder sampling
            
        Returns:
            Dictionary containing:
            - pos: Sampled points
            - point_idx: Indices of sampled points
        """
        # Load point cloud
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        
        # Normalize points
        mean = points.mean(0, keepdim=True)
        points = points - mean

        # Create KDTree for sampling
        kdtree = KDTree(points, leaf_size=40)
        
        # Create cylinder sampler
        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)
        
        # Sample points
        data_dict = {'pos': torch.from_numpy(points.astype(np.float32))}
        sampled_data = cylinder_sampler(kdtree, data_dict)
        
        return {
            'pos': np.asarray(sampled_data['pos']),
            'point_idx': sampled_data['point_idx']
        }
