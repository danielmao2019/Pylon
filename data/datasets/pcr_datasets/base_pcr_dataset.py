from typing import Tuple, Dict, Any, List, Optional
import os
import glob
import numpy as np
import torch
import multiprocessing
from functools import partial
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.sampling import GridSampling3D
from utils.point_cloud_ops import get_correspondences, apply_transform
from utils.point_cloud_ops.grid_sampling import grid_sampling
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

    # Create source point cloud with transformed points
    src_pc_transformed = {'pos': transformed_src_points}

    # Define shifts for partial overlap case
    voxel_size = grid_sampling.size
    shift_amount = voxel_size / 2  # Shift by half the voxel size
    
    # Create a list of target point clouds to process
    # For full overlap (overlap >= 1.0), we only use the original target
    # For partial overlap (overlap < 1.0), we use 6 shifted targets
    target_pcs = []
    
    if overlap >= 1.0:
        # For full overlap, only use the original target
        target_pcs.append({'pos': tgt_points, 'shift': [0, 0, 0]})
    else:
        # For partial overlap, use 6 shifted targets
        shifts = [
            [shift_amount, 0, 0], [-shift_amount, 0, 0],
            [0, shift_amount, 0], [0, -shift_amount, 0],
            [0, 0, shift_amount], [0, 0, -shift_amount]
        ]
        
        for shift in shifts:
            # Apply shift to target points
            shifted_tgt_points = tgt_points.clone()
            shifted_tgt_points[:, 0] += shift[0]
            shifted_tgt_points[:, 1] += shift[1]
            shifted_tgt_points[:, 2] += shift[2]
            
            # Add shifted target to the list
            target_pcs.append({'pos': shifted_tgt_points, 'shift': shift})
    
    datapoints = []
    # Process each target point cloud
    for target_pc in target_pcs:
        # Apply grid sampling to the union of transformed source and target
        mean = torch.cat([src_pc_transformed['pos'], target_pc['pos']], dim=0).mean(0, keepdim=True)
        src_voxels, tgt_voxels = grid_sampling([src_pc_transformed-mean, target_pc['pos']-mean], grid_sampling.size)
        
        # Process each pair of voxels
        for src_voxel, tgt_voxel in zip(src_voxels, tgt_voxels):
            # Get indices for source and target voxels
            src_indices = src_voxel['indices']
            tgt_indices = tgt_voxel['indices']
            
            # Skip if either source or target has too few or too many points
            if len(src_indices) < min_points or len(tgt_indices) < min_points:
                continue
                
            # For partial overlap case, check if the overlap ratio is within the desired range
            if overlap < 1.0:
                # Calculate overlap ratio
                src_points_in_voxel = transformed_src_points[src_indices]
                tgt_points_in_voxel = target_pc['pos'][tgt_indices]
                
                # Find points that are close to each other
                overlap_threshold = voxel_size / 4  # Points within this distance are considered overlapping
                
                # Count source points that are close to any target point
                src_overlapping = 0
                for src_point in src_points_in_voxel:
                    # Find points in the target voxel that are close to this source point
                    distances = torch.norm(tgt_points_in_voxel - src_point, dim=1)
                    close_points = torch.where(distances < overlap_threshold)[0]
                    if len(close_points) > 0:
                        src_overlapping += 1
                
                # Count target points that are close to any source point
                tgt_overlapping = 0
                for tgt_point in tgt_points_in_voxel:
                    # Find points in the source voxel that are close to this target point
                    distances = torch.norm(src_points_in_voxel - tgt_point, dim=1)
                    close_points = torch.where(distances < overlap_threshold)[0]
                    if len(close_points) > 0:
                        tgt_overlapping += 1
                
                # Calculate total overlapping points (union of both sets)
                total_overlapping = src_overlapping + tgt_overlapping
                total_points = len(src_indices) + len(tgt_indices)
                
                # Calculate overlap ratio
                overlap_ratio = total_overlapping / total_points if total_points > 0 else 0
                
                # Skip if overlap ratio is not within the desired range (±10%)
                if abs(overlap_ratio - overlap) > 0.1:
                    continue
            
            if len(src_indices) > max_points:
                src_indices = src_indices[torch.randperm(len(src_indices))[:max_points]]
                
            if len(tgt_indices) > max_points:
                tgt_indices = tgt_indices[torch.randperm(len(tgt_indices))[:max_points]]
            
            # Get the original source points (not transformed)
            src_cluster_points = src_points[src_indices]
            tgt_cluster_points = tgt_points[tgt_indices]
            
            datapoint = {
                'src_points': src_cluster_points,
                'src_indices': src_indices,
                'src_path': src_path,
                'tgt_points': tgt_cluster_points,
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
