from typing import Tuple, Dict, Any
import os
import glob
import numpy as np
import torch
import multiprocessing
from functools import partial
from data.datasets.base_dataset import BaseDataset
from utils.torch_points3d import GridSampling3D
from utils.io import load_point_cloud
from utils.point_cloud_ops import get_correspondences


def process_single_point_cloud(filepath: str, grid_sampling: GridSampling3D) -> list:
    """Process a single point cloud file and return voxel data."""
    # Load point cloud using our utility
    points = load_point_cloud(filepath)[:, :3]  # Only take XYZ coordinates
    points = points.float()

    # Normalize points
    mean = points.mean(0, keepdim=True)
    points = points - mean

    # Grid sample to get point indices for each voxel
    data_dict = {'pos': points}
    sampled_data = grid_sampling(data_dict)

    # Get unique clusters and their points
    cluster_indices = sampled_data['point_indices']  # Shape: (N,) - cluster ID for each point
    unique_clusters = torch.unique(cluster_indices)

    # For each cluster, create voxel data
    voxel_data_list = []
    for cluster_id in unique_clusters:
        cluster_point_indices = torch.where(cluster_indices == cluster_id)[0]
        if len(cluster_point_indices) > 0:  # Only add if cluster has points
            voxel_data = {
                'indices': cluster_point_indices,
                'points': points[cluster_point_indices],
                'filepath': filepath
            }
            voxel_data_list.append(voxel_data)

    return voxel_data_list


def process_single_point_cloud(filepath: str, grid_sampling: GridSampling3D, min_points: int) -> list:
    """Process a single point cloud file and return voxel data."""
    # Load point cloud using our utility
    points = load_point_cloud(filepath)[:, :3]  # Only take XYZ coordinates
    points = points.float()

    # Normalize points
    mean = points.mean(0, keepdim=True)
    points = points - mean

    # Grid sample to get point indices for each voxel
    data_dict = {'pos': points}
    sampled_data = grid_sampling(data_dict)

    # Get unique clusters and their points
    cluster_indices = sampled_data['point_indices']  # Shape: (N,) - cluster ID for each point
    unique_clusters = torch.unique(cluster_indices)

    # For each cluster, create voxel data
    voxel_data_list = []
    for cluster_id in unique_clusters:
        cluster_point_indices = torch.where(cluster_indices == cluster_id)[0]
        if len(cluster_point_indices) >= min_points:  # Only add if cluster has points
            voxel_data = {
                'indices': cluster_point_indices,
                'points': points[cluster_point_indices],
                'filepath': filepath
            }
            voxel_data_list.append(voxel_data)

    return voxel_data_list


def save_voxel_data(args):
    """Save a single voxel data to cache."""
    i, voxel_data, cache_dir = args
    torch.save(voxel_data, os.path.join(cache_dir, f'voxel_{i}.pt'))


class SynthPCRDataset(BaseDataset):
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': None, 'val': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        rot_mag: float = 45.0,
        trans_mag: float = 0.5,
        voxel_size: float = 50.0,
        min_points: int = 128,
        matching_radius: float = 0.1,  # Added matching radius parameter
        **kwargs,
    ) -> None:
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self._voxel_size = voxel_size
        self._min_points = min_points
        self.matching_radius = matching_radius
        self._grid_sampling = GridSampling3D(size=voxel_size)
        super(SynthPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self):
        """Initialize dataset annotations."""
        # Get file paths
        self.file_paths = sorted(glob.glob(os.path.join(self.data_root, '*.ply')))
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), 'voxel_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")

        # Check if cache exists
        if os.path.exists(os.path.join(self.cache_dir, 'voxel_0.pt')):
            # Load all voxel files
            voxel_files = sorted(glob.glob(os.path.join(self.cache_dir, 'voxel_*.pt')))
            self.annotations = voxel_files
            print(f"Loaded {len(voxel_files)} cached voxels")
        else:
            # Process point clouds in parallel
            # Use number of CPU cores minus 1 to leave one core free for system
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Processing point clouds using {num_workers} workers...")

            # Create a partial function with the grid_sampling parameter
            process_func = partial(process_single_point_cloud, grid_sampling=self._grid_sampling, min_points=self._min_points)

            # Use multiprocessing to process files in parallel with chunksize for better performance
            with multiprocessing.Pool(num_workers) as pool:
                # Use chunksize=1 for better load balancing with varying file sizes
                results = pool.map(process_func, self.file_paths, chunksize=1)

            # Flatten the results list
            self.annotations = [voxel for sublist in results for voxel in sublist]

            # Save voxels to cache in parallel
            print(f"Saving {len(self.annotations)} voxels to cache...")
            save_args = [(i, voxel_data, self.cache_dir) for i, voxel_data in enumerate(self.annotations)]
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(save_voxel_data, save_args, chunksize=1)
            print(f"Created and cached {len(self.annotations)} voxels")

        # Split annotations into train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(self.annotations))
        train_idx = int(0.7 * len(indices))
        val_idx = int(0.85 * len(indices))  # 70% + 15%

        if self.split == 'train':
            select_indices = indices[:train_idx]
        elif self.split == 'val':
            select_indices = indices[train_idx:val_idx]
        else:  # test
            select_indices = indices[val_idx:]

        # Select annotations for current split
        self.annotations = [self.annotations[i] for i in select_indices]

        # Update dataset size
        self.DATASET_SIZE[self.split] = len(self.annotations)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint using point indices and generate synthetic pair."""
        # Get voxel data
        voxel_data = self.annotations[idx]

        # If annotations are filepaths, load the data
        if isinstance(voxel_data, str):
            voxel_data = torch.load(voxel_data)

        src_points = voxel_data['points']
        point_indices = voxel_data['indices']
        filepath = voxel_data['filepath']

        assert point_indices.ndim == 1 and point_indices.shape[0] > 0, f"{point_indices.shape=}"

        # Generate random transformation
        rot_mag_rad = np.radians(self.rot_mag)

        # Generate a random axis of rotation
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Generate random angle within the specified range
        angle = torch.empty(1).uniform_(-rot_mag_rad, rot_mag_rad)

        # Create rotation matrix using axis-angle representation (Rodrigues' formula)
        K = torch.tensor([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]], dtype=torch.float32)
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        # Generate random translation
        # Generate random direction (unit vector)
        trans_dir = torch.randn(3, device=src_points.device)
        trans_dir = trans_dir / torch.norm(trans_dir)

        # Generate random magnitude within limit
        trans_mag = torch.empty(1, device=src_points.device).uniform_(0, self.trans_mag)

        # Compute final translation vector
        trans = trans_dir * trans_mag

        # Create 4x4 transformation matrix
        transform = torch.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = trans

        # Apply transformation to create target point cloud
        tgt_points = (R @ src_points.T).T + transform[:3, 3]

        # Find correspondences between source and target point clouds
        correspondences = get_correspondences(
            src_points,
            tgt_points,
            transform,
            self.matching_radius
        )

        inputs = {
            'src_pc': {
                'pos': src_points,
                'feat': torch.ones((src_points.shape[0], 1), dtype=torch.float32),
            },
            'tgt_pc': {
                'pos': tgt_points,
                'feat': torch.ones((tgt_points.shape[0], 1), dtype=torch.float32),
            },
            'correspondences': correspondences,
        }

        labels = {
            'transform': transform,
        }

        meta_info = {
            'idx': idx,
            'point_indices': point_indices,
            'filepath': filepath,
        }

        return inputs, labels, meta_info
