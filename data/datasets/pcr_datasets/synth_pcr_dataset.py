import os
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.torch_points3d import GridSampling3D
from utils.io import load_point_cloud


class SynthPCRDataset(BaseDataset):
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': None, 'val': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        rot_mag: float = 45.0,
        trans_mag: float = 0.5,
        voxel_size: float = 50.0,
        **kwargs,
    ) -> None:
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self._voxel_size = voxel_size
        self._grid_sampling = GridSampling3D(size=voxel_size)
        super(SynthPCRDataset, self).__init__(**kwargs)

    def _prepare_all_centers(self):
        """Prepare all voxel centers by grid sampling each point cloud."""
        all_indices = []
        for filepath in self.file_paths:
            # Load point cloud using our utility
            points = load_point_cloud(filepath)[:, :3]  # Only take XYZ coordinates
            points = points.float()

            # Normalize points
            mean = points.mean(0, keepdim=True)
            points = points - mean

            # Grid sample to get point indices for each voxel
            data_dict = {'pos': points}
            sampled_data = self._grid_sampling(data_dict)

            # Get unique clusters and their points
            cluster_indices = sampled_data['point_indices']  # Shape: (N,) - cluster ID for each point
            unique_clusters = torch.unique(cluster_indices)

            # For each cluster, get the indices of points belonging to it
            for cluster_id in unique_clusters:
                cluster_point_indices = torch.where(cluster_indices == cluster_id)[0]
                if len(cluster_point_indices) > 0:  # Only add if cluster has points
                    all_indices.append(cluster_point_indices)

            print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")
            print(f"Partitioned point cloud into {len(unique_clusters)} voxels.")

        print(f"Total number of voxels across all point clouds: {len(all_indices)}")
        return all_indices

    def _init_annotations(self):
        """Initialize dataset annotations."""
        # Get file paths
        self.file_paths = []
        for file in os.listdir(self.data_root):
            if file.endswith('.ply'):
                self.file_paths.append(os.path.join(self.data_root, file))
        self.file_paths = self.file_paths[:1]
        print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")

        # Get all voxel point indices
        all_indices = self._prepare_all_centers()
        print(f"Partitioned {len(self.file_paths)} point clouds into {len(all_indices)} voxels in total.")

        # Split indices into train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(all_indices))
        train_idx = int(0.7 * len(indices))
        val_idx = int(0.85 * len(indices))  # 70% + 15%

        if self.split == 'train':
            select_indices = indices[:train_idx]
        elif self.split == 'val':
            select_indices = indices[train_idx:val_idx]
        else:  # test
            select_indices = indices[val_idx:]

        # Store point indices for current split
        self.annotations = [all_indices[i] for i in select_indices]

        # Update dataset size
        self.DATASET_SIZE[self.split] = len(self.annotations)

    def _load_datapoint(self, idx):
        """Load a datapoint using point indices and generate synthetic pair."""
        # Get point indices for this voxel
        point_indices = self.annotations[idx]
        assert point_indices.ndim == 1 and point_indices.shape[0] > 0, f"{point_indices.shape=}"

        # Load point cloud using our utility
        points = load_point_cloud(self.file_paths[0])[:, :3]  # Only take XYZ coordinates
        points = points.float()

        # Normalize points
        mean = points.mean(0, keepdim=True)
        points = points - mean

        # Get points in this voxel
        src_points = points[point_indices, :]

        # Generate random transformation
        rot_mag_rad = np.radians(self.rot_mag)
        
        # Generate a random axis of rotation
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector
        
        # Generate random angle within the specified range
        angle = torch.empty(1).uniform_(-rot_mag_rad, rot_mag_rad).item()
        
        # Create rotation matrix using axis-angle representation (Rodrigues' formula)
        K = torch.tensor([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]], dtype=torch.float32)
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        
        # Generate random translation
        trans = torch.empty(3).uniform_(-self.trans_mag, self.trans_mag)

        # Create 4x4 transformation matrix
        transform = torch.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = trans

        # Apply transformation to create target point cloud
        tgt_points = (R @ src_points.T).T + transform[:3, 3]

        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32),
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32),
            },
        }

        labels = {
            'transform': transform,
        }

        meta_info = {
            'idx': idx,
            'point_indices': point_indices,
            'filepath': self.file_paths[0],
        }

        return inputs, labels, meta_info
