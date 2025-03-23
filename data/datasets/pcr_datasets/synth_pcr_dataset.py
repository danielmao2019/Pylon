import os
import numpy as np
import torch
import open3d as o3d
from data.datasets.base_dataset import BaseDataset
from scipy.spatial import KDTree
from data.datasets.cylinder_sampling import CylinderSampling


class SynthPCRDataset(BaseDataset):
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {'train': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(self, rot_mag=45.0, trans_mag=0.5, **kwargs):
        assert isinstance(rot_mag, float)
        assert isinstance(trans_mag, float)
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        super(SynthPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self):
        """Initialize dataset annotations"""
        self.annotations = []
        for file in os.listdir(self.data_root):
            if file.endswith('.ply'):
                self.annotations.append(os.path.join(self.data_root, file))

        if self.split in ['train', 'test']:
            np.random.seed(42)
            indices = np.random.permutation(len(self.annotations))
            split_idx = int(0.8 * len(self.annotations))
            if self.split == 'train':
                self.annotations = [self.annotations[i] for i in indices[:split_idx]]
            else:
                self.annotations = [self.annotations[i] for i in indices[split_idx:]]

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
        kdtree = KDTree(points)
        
        # Create cylinder sampler
        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)
        
        # Sample points
        data_dict = {'pos': torch.from_numpy(points.astype(np.float32))}
        sampled_data = cylinder_sampler(kdtree, data_dict)
        
        return {
            'pos': np.asarray(sampled_data['pos']),
            'point_idx': sampled_data['point_idx']
        }
