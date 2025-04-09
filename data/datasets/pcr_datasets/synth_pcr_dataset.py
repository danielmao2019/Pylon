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


def process_single_point_cloud(filepath: str, grid_sampling: GridSampling3D, min_points: int, max_points: int, overlap: float = 1.0) -> list:
    """Process a single point cloud file and return voxel data.
    
    Args:
        filepath: Path to the point cloud file
        grid_sampling: Grid sampling object for voxelization
        min_points: Minimum number of points in a voxel
        max_points: Maximum number of points in a voxel
        overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
    """
    # Load point cloud using our utility
    pc = load_point_cloud(filepath)  # Only take XYZ coordinates
    assert isinstance(pc, dict)
    assert pc.keys() >= {'pos'}
    points = pc['pos'].float()

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
    
    # If overlap is 1.0, use the original implementation
    if overlap >= 1.0:
        for cluster_id in unique_clusters:
            cluster_point_indices = torch.where(cluster_indices == cluster_id)[0]
            if len(cluster_point_indices) >= min_points:  # Only add if cluster has points
                # Random sampling if cluster exceeds max_points
                if len(cluster_point_indices) > max_points:
                    # Use random permutation to randomly sample points
                    perm = torch.randperm(len(cluster_point_indices))
                    cluster_point_indices = cluster_point_indices[perm[:max_points]]

                voxel_data = {
                    'indices': cluster_point_indices,
                    'points': points[cluster_point_indices],
                    'filepath': filepath
                }
                voxel_data_list.append(voxel_data)
    else:
        # For partial overlap, create shifted voxelizations
        voxel_size = grid_sampling.size
        shift_amount = voxel_size / 2  # Shift by half the voxel size
        
        # Create 6 shifted grid samplings (one in each direction)
        shifted_grids = []
        shifts = [
            [shift_amount, 0, 0], [-shift_amount, 0, 0],
            [0, shift_amount, 0], [0, -shift_amount, 0],
            [0, 0, shift_amount], [0, 0, -shift_amount]
        ]
        
        for shift in shifts:
            # Create a new grid sampling with the shift
            shifted_grid = GridSampling3D(size=voxel_size)
            # Apply shift to the points before sampling
            shifted_points = points.clone()
            shifted_points[:, 0] += shift[0]
            shifted_points[:, 1] += shift[1]
            shifted_points[:, 2] += shift[2]
            
            shifted_data = shifted_grid({'pos': shifted_points})
            shifted_grids.append((shifted_data['point_indices'], shift))
        
        # For each original voxel, find overlapping voxels with desired IoU
        for cluster_id in unique_clusters:
            cluster_point_indices = torch.where(cluster_indices == cluster_id)[0]
            if len(cluster_point_indices) < min_points:
                continue
                
            # Get points in this voxel
            voxel_points = points[cluster_point_indices]
            
            # Check each shifted grid for overlapping voxels
            for shifted_indices, shift in shifted_grids:
                # Find the corresponding cluster in the shifted grid
                # We need to find which cluster in the shifted grid contains points that overlap with our voxel
                
                # Get points in the shifted grid
                shifted_points = points.clone()
                shifted_points[:, 0] += shift[0]
                shifted_points[:, 1] += shift[1]
                shifted_points[:, 2] += shift[2]
                
                # Find which points in our voxel are in the shifted grid
                # We'll use a simple distance-based approach
                # For each point in our voxel, find if there's a point in the shifted grid within a small distance
                overlap_threshold = voxel_size / 4  # Points within this distance are considered overlapping
                
                # For each point in our voxel, check if there's a point in the shifted grid within the threshold
                overlapping_points = []
                for i, point in enumerate(voxel_points):
                    # Find points in the shifted grid that are close to this point
                    distances = torch.norm(shifted_points - point, dim=1)
                    close_points = torch.where(distances < overlap_threshold)[0]
                    if len(close_points) > 0:
                        overlapping_points.append(i)
                
                # If we have enough overlapping points, create a pair
                if len(overlapping_points) >= min_points:
                    # Get the cluster ID in the shifted grid that contains these overlapping points
                    overlapping_indices = cluster_point_indices[overlapping_points]
                    overlapping_shifted_indices = shifted_indices[overlapping_indices]
                    unique_shifted_clusters = torch.unique(overlapping_shifted_indices)
                    
                    # For each overlapping cluster in the shifted grid
                    for shifted_cluster_id in unique_shifted_clusters:
                        # Get points in this shifted cluster
                        shifted_cluster_indices = torch.where(shifted_indices == shifted_cluster_id)[0]
                        if len(shifted_cluster_indices) < min_points:
                            continue
                        
                        # Calculate IoU
                        # Points in the intersection are those that are in both clusters
                        intersection_points = set(overlapping_points)
                        intersection_points.intersection_update(set(shifted_cluster_indices.tolist()))
                        
                        # Union is all points in both clusters
                        union_points = set(cluster_point_indices.tolist())
                        union_points.update(shifted_cluster_indices.tolist())
                        
                        # Calculate IoU
                        if len(union_points) > 0:
                            iou = len(intersection_points) / len(union_points)
                            
                            # Check if IoU is in the desired range (overlap ± 10%)
                            if abs(iou - overlap) <= 0.1:
                                # Create a pair of voxels
                                # Source voxel
                                src_indices = cluster_point_indices
                                if len(src_indices) > max_points:
                                    perm = torch.randperm(len(src_indices))
                                    src_indices = src_indices[perm[:max_points]]
                                
                                # Target voxel (from shifted grid)
                                tgt_indices = shifted_cluster_indices
                                if len(tgt_indices) > max_points:
                                    perm = torch.randperm(len(tgt_indices))
                                    tgt_indices = tgt_indices[perm[:max_points]]
                                
                                # Create voxel data for both source and target
                                src_voxel_data = {
                                    'indices': src_indices,
                                    'points': points[src_indices],
                                    'filepath': filepath,
                                    'is_source': True
                                }
                                
                                tgt_voxel_data = {
                                    'indices': tgt_indices,
                                    'points': shifted_points[tgt_indices],
                                    'filepath': filepath,
                                    'is_source': False
                                }
                                
                                voxel_data_list.append((src_voxel_data, tgt_voxel_data))

    return voxel_data_list


def save_voxel_data(args):
    """Save a single voxel data to cache."""
    i, voxel_data, cache_dir = args
    # Check if voxel_data is a tuple (for partial overlap) or a single dict (for full overlap)
    if isinstance(voxel_data, tuple):
        # For partial overlap, save both source and target voxels
        src_voxel, tgt_voxel = voxel_data
        torch.save((src_voxel, tgt_voxel), os.path.join(cache_dir, f'voxel_{i}.pt'))
    else:
        # For full overlap, save as before
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
        min_points: int = 256,
        max_points: int = 8192,
        matching_radius: float = 0.1,
        overlap: float = 1.0,
        **kwargs,
    ) -> None:
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self._voxel_size = voxel_size
        self._min_points = min_points
        self._max_points = max_points
        self.matching_radius = matching_radius
        self.overlap = overlap
        self._grid_sampling = GridSampling3D(size=voxel_size)
        super(SynthPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self):
        """Initialize dataset annotations."""
        # Get file paths
        self.file_paths = sorted(glob.glob(os.path.join(self.data_root, '*.las')))
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), 'voxel_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")

        # Check if cache exists
        voxel_files = sorted(glob.glob(os.path.join(self.cache_dir, 'voxel_*.pt')))
        if len(voxel_files) > 0:
            # Load all voxel files
            self.annotations = voxel_files
            print(f"Loaded {len(voxel_files)} cached voxels")
        else:
            # Process point clouds in parallel
            # Use number of CPU cores minus 1 to leave one core free for system
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Processing point clouds using {num_workers} workers...")

            # Create a partial function with the grid_sampling parameter
            process_func = partial(
                process_single_point_cloud,
                grid_sampling=self._grid_sampling,
                min_points=self._min_points,
                max_points=self._max_points,
                overlap=self.overlap,
            )

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

        # Check if voxel_data is a tuple (for partial overlap) or a single dict (for full overlap)
        if isinstance(voxel_data, tuple):
            # For partial overlap, we already have source and target voxels
            src_voxel, tgt_voxel = voxel_data
            src_points = src_voxel['points']
            tgt_points = tgt_voxel['points']
            point_indices = src_voxel['indices']
            filepath = src_voxel['filepath']
            
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
            tgt_points = (R @ tgt_points.T).T + transform[:3, 3]
        else:
            # For full overlap, use the original implementation
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
