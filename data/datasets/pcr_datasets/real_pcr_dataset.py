from typing import List, Dict, Any, Tuple
from functools import partial
import os
import multiprocessing
import glob
import json
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.io import load_point_cloud
from utils.point_cloud_ops import apply_transform
from utils.torch_points3d import GridSampling3D
from utils.point_cloud_ops import get_correspondences


def process_single_point_cloud(src_path: str, tgt_path: str, gt_transform: torch.Tensor, grid_sampling, min_points, max_points, overlap: float = 1.0) -> List[Dict[str, Any]]:
    """Process a pair of point clouds and return datapoints with controlled overlap.
    
    Args:
        src_path: Path to the source point cloud file
        tgt_path: Path to the target point cloud file
        gt_transform: Ground truth transformation from source to target
        grid_sampling: Grid sampling object for voxelization
        min_points: Minimum number of points in a voxel
        max_points: Maximum number of points in a voxel
        overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
    """
    src_points = load_point_cloud(src_path)['pos']
    tgt_points = load_point_cloud(tgt_path)['pos']
    transformed_src_points = apply_transform(src_points, gt_transform)

    # If overlap is 1.0, use the original implementation
    if overlap >= 1.0:
        # Combine source and target points
        union_points = torch.cat([transformed_src_points, tgt_points], dim=0)
        union_points = union_points.float()

        # Center the points
        mean = union_points.mean(0, keepdim=True)
        union_points = union_points - mean

        # Apply grid sampling to get clusters
        sampled_data = grid_sampling({'pos': union_points})
        cluster_indices = sampled_data['point_indices']

        # Create a mask to distinguish source and target points
        # First N points are from source, rest are from target
        src_count = transformed_src_points.shape[0]
        src_mask = torch.zeros(union_points.shape[0], dtype=torch.bool)
        src_mask[:src_count] = True
        tgt_mask = ~src_mask

        datapoints = []
        # Process each unique cluster
        for cluster_id in torch.unique(cluster_indices):
            # Get points in this cluster
            cluster_mask = cluster_indices == cluster_id
            src_cluster_mask = cluster_mask & src_mask
            tgt_cluster_mask = cluster_mask & tgt_mask

            src_cluster_indices = torch.where(src_cluster_mask)[0]
            tgt_cluster_indices = torch.where(tgt_cluster_mask)[0] - src_count

            # Skip if either source or target has too few or too many points
            if len(src_cluster_indices) < min_points or len(tgt_cluster_indices) < min_points:
                continue

            if len(src_cluster_indices) > max_points:
                src_cluster_indices = src_cluster_indices[torch.randperm(len(src_cluster_indices))[:max_points]]

            if len(tgt_cluster_indices) > max_points:
                tgt_cluster_indices = tgt_cluster_indices[torch.randperm(len(tgt_cluster_indices))[:max_points]]

            src_cluster_points = src_points[src_cluster_indices]
            tgt_cluster_points = tgt_points[tgt_cluster_indices]

            datapoint = {
                'src_points': src_cluster_points,
                'src_indices': src_cluster_indices,
                'src_path': src_path,
                'tgt_points': tgt_cluster_points,
                'tgt_indices': tgt_cluster_indices,
                'tgt_path': tgt_path,
                'transform': gt_transform,
            }
            datapoints.append(datapoint)
        return datapoints
    else:
        # For partial overlap, create shifted voxelizations
        voxel_size = grid_sampling.size
        shift_amount = voxel_size / 2  # Shift by half the voxel size
        
        # Center the points
        src_mean = transformed_src_points.mean(0, keepdim=True)
        tgt_mean = tgt_points.mean(0, keepdim=True)
        transformed_src_points = transformed_src_points - src_mean
        tgt_points = tgt_points - tgt_mean
        
        # Apply grid sampling to source points
        src_sampled_data = grid_sampling({'pos': transformed_src_points})
        src_cluster_indices = src_sampled_data['point_indices']
        src_unique_clusters = torch.unique(src_cluster_indices)
        
        # Create 6 shifted grid samplings for target points (one in each direction)
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
            shifted_points = tgt_points.clone()
            shifted_points[:, 0] += shift[0]
            shifted_points[:, 1] += shift[1]
            shifted_points[:, 2] += shift[2]
            
            shifted_data = shifted_grid({'pos': shifted_points})
            shifted_grids.append((shifted_data['point_indices'], shift))
        
        datapoints = []
        # For each source voxel, find overlapping target voxels with desired IoU
        for src_cluster_id in src_unique_clusters:
            src_cluster_mask = src_cluster_indices == src_cluster_id
            src_cluster_indices_full = torch.where(src_cluster_mask)[0]
            
            if len(src_cluster_indices_full) < min_points:
                continue
                
            # Get points in this source voxel
            src_voxel_points = transformed_src_points[src_cluster_indices_full]
            
            # Check each shifted grid for overlapping voxels
            for tgt_indices, shift in shifted_grids:
                # Apply shift to target points
                shifted_tgt_points = tgt_points.clone()
                shifted_tgt_points[:, 0] += shift[0]
                shifted_tgt_points[:, 1] += shift[1]
                shifted_tgt_points[:, 2] += shift[2]
                
                # Find which points in our source voxel are in the shifted target grid
                # We'll use a simple distance-based approach
                overlap_threshold = voxel_size / 4  # Points within this distance are considered overlapping
                
                # For each point in our source voxel, check if there's a point in the shifted target grid within the threshold
                overlapping_points = []
                for i, point in enumerate(src_voxel_points):
                    # Find points in the shifted target grid that are close to this point
                    distances = torch.norm(shifted_tgt_points - point, dim=1)
                    close_points = torch.where(distances < overlap_threshold)[0]
                    if len(close_points) > 0:
                        overlapping_points.append(i)
                
                # If we have enough overlapping points, create a pair
                if len(overlapping_points) >= min_points:
                    # Get the cluster ID in the shifted target grid that contains these overlapping points
                    overlapping_indices = src_cluster_indices_full[overlapping_points]
                    overlapping_tgt_indices = tgt_indices[overlapping_indices]
                    unique_tgt_clusters = torch.unique(overlapping_tgt_indices)
                    
                    # For each overlapping cluster in the shifted target grid
                    for tgt_cluster_id in unique_tgt_clusters:
                        # Get points in this target cluster
                        tgt_cluster_mask = tgt_indices == tgt_cluster_id
                        tgt_cluster_indices_full = torch.where(tgt_cluster_mask)[0]
                        
                        if len(tgt_cluster_indices_full) < min_points:
                            continue
                        
                        # Calculate IoU
                        # Points in the intersection are those that are in both clusters
                        intersection_points = set(overlapping_points)
                        intersection_points.intersection_update(set(tgt_cluster_indices_full.tolist()))
                        
                        # Union is all points in both clusters
                        union_points = set(src_cluster_indices_full.tolist())
                        union_points.update(tgt_cluster_indices_full.tolist())
                        
                        # Calculate IoU
                        if len(union_points) > 0:
                            iou = len(intersection_points) / len(union_points)
                            
                            # Check if IoU is in the desired range (overlap ± 10%)
                            if abs(iou - overlap) <= 0.1:
                                # Create a pair of voxels
                                # Source voxel
                                src_indices = src_cluster_indices_full
                                if len(src_indices) > max_points:
                                    perm = torch.randperm(len(src_indices))
                                    src_indices = src_indices[perm[:max_points]]
                                
                                # Target voxel
                                tgt_indices = tgt_cluster_indices_full
                                if len(tgt_indices) > max_points:
                                    perm = torch.randperm(len(tgt_indices))
                                    tgt_indices = tgt_indices[perm[:max_points]]
                                
                                # Create datapoint
                                datapoint = {
                                    'src_points': src_points[src_indices],
                                    'src_indices': src_indices,
                                    'src_path': src_path,
                                    'tgt_points': tgt_points[tgt_indices],
                                    'tgt_indices': tgt_indices,
                                    'tgt_path': tgt_path,
                                    'transform': gt_transform,
                                }
                                datapoints.append(datapoint)
        
        return datapoints


def save_datapoint(args):
    """Save a datapoint to disk.

    Args:
        args: Tuple containing (index, datapoint, cache_dir)
    """
    idx, datapoint, cache_dir = args
    cache_path = os.path.join(cache_dir, f'datapoint_{idx:06d}.pt')
    torch.save(datapoint, cache_path)
    return cache_path


class RealPCRDataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': None, 'val': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        gt_transforms: str,
        voxel_size: float = 10.0,
        min_points: int = 256,
        max_points: int = 8192,
        matching_radius: float = 0.1,
        overlap: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the dataset.

        Args:
            gt_transforms: Path to JSON file containing ground truth transformations
            voxel_size: Size of voxel cells for sampling (default: 10.0)
            min_points: Minimum number of points in a cluster (default: 256)
            max_points: Maximum number of points in a cluster (default: 8192)
            matching_radius: Radius for finding correspondences (default: 0.1)
            overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
            **kwargs: Additional arguments passed to BaseDataset
        """
        self.gt_transforms = gt_transforms
        self._voxel_size = voxel_size
        self._min_points = min_points
        self._max_points = max_points
        self.matching_radius = matching_radius
        self.overlap = overlap
        self._grid_sampling = GridSampling3D(size=voxel_size)
        super(RealPCRDataset, self).__init__(**kwargs)

    def _init_annotations(self) -> None:
        self.file_paths = sorted(glob.glob(os.path.join(self.data_root, '*.las')))
        with open(self.gt_transforms, 'r') as f:
            self.gt_transforms: List[Dict[str, Any]] = json.load(f)
        assert len(self.file_paths) == len(self.gt_transforms), \
            f"{self.file_paths=}, {[gt['filepath'] for gt in self.gt_transforms]=}"
        assert all(isinstance(transform, dict) for transform in self.gt_transforms)
        assert all(transform.keys() == {'filepath', 'transform'} for transform in self.gt_transforms)
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), 'real_pcr_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_files = sorted(glob.glob(os.path.join(self.cache_dir, 'datapoint_*.pt')))
        if len(cache_files) > 0:
            self.annotations = cache_files
            print(f"Loaded {len(cache_files)} cached datapoints.")
        else:
            print(f"No cached datapoints found in {self.cache_dir}")
            num_workers = max(1, multiprocessing.cpu_count() - 1)
            print(f"Processing point clouds using {num_workers} workers...")

            process_func = partial(
                process_single_point_cloud,
                grid_sampling=self._grid_sampling,
                min_points=self._min_points,
                max_points=self._max_points,
                overlap=self.overlap,
            )

            # Use multiprocessing to process files in parallel with chunksize for better performance
            with multiprocessing.Pool(num_workers) as pool:
                # Create arguments for each file
                process_args = []
                for transform_data in self.gt_transforms[1:]:
                    src_path = os.path.join(self.data_root, transform_data['filepath'])
                    tgt_path = self.file_paths[0]
                    gt_transform = torch.tensor(transform_data['transform'], dtype=torch.float32)
                    process_args.append((src_path, tgt_path, gt_transform))

                # Process files in parallel
                results = pool.starmap(process_func, process_args, chunksize=1)

            # Flatten the results list
            self.annotations = [datapoint for sublist in results for datapoint in sublist]

            # Save datapoints to cache in parallel
            print(f"Saving {len(self.annotations)} datapoints to cache...")
            save_args = [(i, datapoint, self.cache_dir) for i, datapoint in enumerate(self.annotations)]
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(save_datapoint, save_args, chunksize=1)
            print(f"Created and cached {len(self.annotations)} datapoints.")

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
        datapoint_cache = self.annotations[idx]

        # If annotations are filepaths, load the data
        if isinstance(datapoint_cache, str):
            datapoint_cache = torch.load(datapoint_cache)

        src_points = datapoint_cache['src_points']
        tgt_points = datapoint_cache['tgt_points']
        gt_transform = datapoint_cache['transform']

        # Find correspondences between source and target point clouds
        correspondences = get_correspondences(
            src_points,
            tgt_points,
            gt_transform,
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
            'transform': gt_transform,
        }

        meta_info = {
            'idx': idx,
            'src_indices': datapoint_cache['src_indices'],
            'src_path': datapoint_cache['src_path'],
            'tgt_indices': datapoint_cache['tgt_indices'],
            'tgt_path': datapoint_cache['tgt_path'],
        }

        return inputs, labels, meta_info
