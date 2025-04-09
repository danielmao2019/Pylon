from typing import Tuple, Dict, Any, List, Optional
import os
import glob
import numpy as np
import torch
import multiprocessing
from functools import partial
from data.datasets.base_dataset import BaseDataset
from utils.torch_points3d import GridSampling3D
from utils.point_cloud_ops import get_correspondences, apply_transform
from utils.io import load_point_cloud


def process_point_cloud_pair(
    src_path: str, tgt_path: str, transform: torch.Tensor, 
    grid_sampling: GridSampling3D, min_points: int, max_points: int,
    overlap: float = 1.0,
) -> List[Dict[str, Any]]:
    """Process a pair of point clouds and return datapoints.
    
    Args:
        src_path: Path to the source point cloud file
        tgt_path: Path to the target point cloud file
        transform: Transformation from source to target
        grid_sampling: Grid sampling object for voxelization
        min_points: Minimum number of points in a voxel
        max_points: Maximum number of points in a voxel
        overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
    """
    # Load source point cloud
    src_pc = load_point_cloud(src_path)
    assert isinstance(src_pc, dict)
    assert src_pc.keys() >= {'pos'}
    src_points = src_pc['pos'].float()
    
    # Load target point cloud
    tgt_pc = load_point_cloud(tgt_path)
    assert isinstance(tgt_pc, dict)
    assert tgt_pc.keys() >= {'pos'}
    tgt_points = tgt_pc['pos'].float()
    
    # Transform source points to align with target
    transformed_src_points = apply_transform(src_points, transform)
    
    # If overlap is 1.0, use the original implementation
    if overlap >= 1.0:
        # Combine source and target points
        union_points = torch.cat([transformed_src_points, tgt_points], dim=0)
        
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
            
            # Get the original source points (not transformed)
            src_cluster_points = src_points[src_cluster_indices]
            tgt_cluster_points = tgt_points[tgt_cluster_indices]
            
            datapoint = {
                'src_points': src_cluster_points,
                'src_indices': src_cluster_indices,
                'src_path': src_path,
                'tgt_points': tgt_cluster_points,
                'tgt_indices': tgt_cluster_indices,
                'tgt_path': tgt_path,
                'transform': transform,
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
                                    'transform': transform,
                                }
                                datapoints.append(datapoint)
        
        return datapoints


def save_datapoint(args):
    """Save a single datapoint to cache.
    
    Args:
        args: Tuple containing (index, datapoint, cache_dir)
    """
    i, datapoint, cache_dir = args
    torch.save(datapoint, os.path.join(cache_dir, f'voxel_{i}.pt'))


class BasePCRDataset(BaseDataset):
    """Base class for point cloud registration datasets.
    
    This class provides common functionality for both synthetic and real point cloud registration datasets.
    """
    # Required class attributes from BaseDataset
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': None, 'val': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None

    def __init__(
        self,
        voxel_size: float = 50.0,
        min_points: int = 256,
        max_points: int = 8192,
        matching_radius: float = 0.1,
        overlap: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            voxel_size: Size of voxel cells for sampling (default: 50.0)
            min_points: Minimum number of points in a cluster (default: 256)
            max_points: Maximum number of points in a cluster (default: 8192)
            matching_radius: Radius for finding correspondences (default: 0.1)
            overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
            **kwargs: Additional arguments passed to BaseDataset
        """
        self._voxel_size = voxel_size
        self._min_points = min_points
        self._max_points = max_points
        self.matching_radius = matching_radius
        self.overlap = overlap
        self._grid_sampling = GridSampling3D(size=voxel_size)
        super(BasePCRDataset, self).__init__(**kwargs)
        
        # Initialize file path pairs and transforms
        self._init_file_pairs()

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.
        
        This method should be overridden by subclasses to set up:
        - self.src_file_paths: List of source file paths
        - self.tgt_file_paths: List of target file paths
        - self.transforms: List of transforms from source to target
        """
        raise NotImplementedError("Subclasses must implement _init_file_pairs")

    def _split_annotations(self, annotations: List[Any]) -> List[Any]:
        """Split annotations into train/val/test sets.
        
        Args:
            annotations: List of annotations to split
            
        Returns:
            List of annotations for the current split
        """
        np.random.seed(42)
        indices = np.random.permutation(len(annotations))
        train_idx = int(0.7 * len(indices))
        val_idx = int(0.85 * len(indices))  # 70% + 15%

        if self.split == 'train':
            select_indices = indices[:train_idx]
        elif self.split == 'val':
            select_indices = indices[train_idx:val_idx]
        else:  # test
            select_indices = indices[val_idx:]

        # Select annotations for current split
        selected_annotations = [annotations[i] for i in select_indices]

        # Update dataset size
        self.DATASET_SIZE[self.split] = len(selected_annotations)
        
        return selected_annotations

    def _init_annotations(self) -> None:
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
                self._process_point_cloud_pair,
                grid_sampling=self._grid_sampling,
                min_points=self._min_points,
                max_points=self._max_points,
                overlap=self.overlap,
            )

            # Use multiprocessing to process files in parallel with chunksize for better performance
            with multiprocessing.Pool(num_workers) as pool:
                # Create arguments for each file pair
                process_args = []
                for src_path, tgt_path, transform in zip(self.src_file_paths, self.tgt_file_paths, self.transforms):
                    process_args.append((src_path, tgt_path, transform))

                # Process files in parallel
                results = pool.starmap(process_func, process_args, chunksize=1)

            # Flatten the results list
            self.annotations = [voxel for sublist in results for voxel in sublist]

            # Save voxels to cache in parallel
            print(f"Saving {len(self.annotations)} voxels to cache...")
            save_args = [(i, voxel_data, self.cache_dir) for i, voxel_data in enumerate(self.annotations)]
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(self._save_voxel_data, save_args, chunksize=1)
            print(f"Created and cached {len(self.annotations)} voxels")

        # Split annotations into train/val/test
        self.annotations = self._split_annotations(self.annotations)

    def _process_point_cloud_pair(self, src_path: str, tgt_path: str, transform: torch.Tensor, 
                                 grid_sampling: GridSampling3D, min_points: int, max_points: int, 
                                 overlap: float = 1.0) -> List[Dict[str, Any]]:
        """Process a pair of point clouds and return datapoints.
        
        This method should be implemented by subclasses to process a pair of point clouds.
        """
        return process_point_cloud_pair(
            src_path, tgt_path, transform, 
            grid_sampling, min_points, max_points,
            overlap
        )

    def _save_voxel_data(self, args) -> None:
        """Save a single voxel data to cache.
        
        Args:
            args: Tuple containing (index, voxel_data, cache_dir)
        """
        save_datapoint(args)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint using point indices."""
        # Get voxel data
        voxel_data = self.annotations[idx]

        # If annotations are filepaths, load the data
        if isinstance(voxel_data, str):
            voxel_data = torch.load(voxel_data)

        # Extract data from voxel_data
        src_points = voxel_data['src_points']
        tgt_points = voxel_data['tgt_points']
        src_indices = voxel_data['src_indices']
        tgt_indices = voxel_data['tgt_indices']
        src_path = voxel_data['src_path']
        tgt_path = voxel_data['tgt_path']
        transform = voxel_data['transform']

        assert src_indices.ndim == 1 and src_indices.shape[0] > 0, f"{src_indices.shape=}"
        assert tgt_indices.ndim == 1 and tgt_indices.shape[0] > 0, f"{tgt_indices.shape=}"

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
            'src_indices': src_indices,
            'tgt_indices': tgt_indices,
            'src_path': src_path,
            'tgt_path': tgt_path,
        }

        return inputs, labels, meta_info
