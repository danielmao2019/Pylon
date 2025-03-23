import os
import numpy as np
import torch
import open3d as o3d
from data.datasets.base_dataset import BaseDataset

class SynthPCRDataset(BaseDataset):
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'test']
    DATASET_SIZE = {'train': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(self,
                 data_root,
                 split='train',
                 rot_mag=45.0,
                 trans_mag=0.5,
                 **kwargs):
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        super().__init__(data_root=data_root, split=split, **kwargs)

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
        """Load a single datapoint."""
        # Load source point cloud
        src_pcd = o3d.io.read_point_cloud(self.annotations[idx])
        src_points = np.asarray(src_pcd.points)
        
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

        # Apply transformation to get target point cloud
        tgt_points = (R @ src_points.T).T + trans

        # Convert to torch tensors
        src_points = torch.from_numpy(src_points.astype(np.float32))
        tgt_points = torch.from_numpy(tgt_points.astype(np.float32))
        transform = torch.from_numpy(transform.astype(np.float32))

        inputs = {
            'src_pc': src_points[None],
            'tgt_pc': tgt_points[None]
        }
        labels = {
            'transform': transform[None]
        }
        meta_info = {
            'filename': self.annotations[idx]
        }

        return inputs, labels, meta_info
