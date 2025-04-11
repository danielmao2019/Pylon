from typing import Tuple, Dict, Any, List
from functools import partial
import copy
import os
import glob
import numpy as np
import torch
import multiprocessing
from data.datasets.base_dataset import BaseDataset
from data.transforms.vision_3d.select import Select
from data.transforms.vision_3d.random_select import RandomSelect
from utils.point_cloud_ops import get_correspondences, apply_transform
from utils.point_cloud_ops.grid_sampling import grid_sampling
from utils.io import load_point_cloud
from utils.point_cloud_ops.set_ops.intersection import compute_pc_iou
from utils.ops import apply_tensor_op


def process_voxel_pair(args):
    """Process a single pair of voxels and return a datapoint if valid.
    
    Args:
        args: Tuple containing (src_voxel, tgt_voxel, transformed_src_pc, src_pc, tgt_pc, 
              src_path, tgt_path, transform, min_points, max_points, overlap, voxel_size)
    
    Returns:
        Datapoint dictionary if valid, None otherwise
    """
    src_voxel, tgt_voxel, transformed_src_pc, src_pc, tgt_pc, src_path, tgt_path, transform, min_points, max_points, overlap, voxel_size = args
    
    if (src_voxel is None) or (tgt_voxel is None):
        return None
    # Skip if either source or target has too few points
    if len(src_voxel['indices']) < min_points or len(tgt_voxel['indices']) < min_points:
        return None

    # For partial overlap case, check if the overlap ratio is within the desired range
    if overlap < 1.0:
        overlap_ratio = compute_pc_iou(
            src_points=Select(indices=src_voxel['indices'])(transformed_src_pc)['pos'],
            tgt_points=Select(indices=tgt_voxel['indices'])(tgt_pc)['pos'],
            radius=voxel_size / 4,
        )
        # Skip if overlap ratio is not within the desired range (±10%)
        if abs(overlap_ratio - overlap) > 0.1:
            return None

    # Apply random sampling if needed
    src_pc_final = Select(src_voxel['indices'])(src_pc)
    if len(src_voxel['indices']) > max_points:
        src_pc_final = RandomSelect(percentage=max_points / len(src_voxel['indices']))(src_pc_final)

    tgt_pc_final = Select(tgt_voxel['indices'])(tgt_pc)
    if len(tgt_voxel['indices']) > max_points:
        tgt_pc_final = RandomSelect(percentage=max_points / len(tgt_voxel['indices']))(tgt_pc_final)

    datapoint = {
        'src_points': src_pc_final['pos'],
        'src_indices': src_pc_final['indices'],
        'src_path': src_path,
        'tgt_points': tgt_pc_final['pos'],
        'tgt_indices': tgt_pc_final['indices'],
        'tgt_path': tgt_path,
        'transform': transform,
    }
    return datapoint


def process_point_cloud_pair(
    src_path: str, tgt_path: str, transform: torch.Tensor,
    voxel_size: float, min_points: int, max_points: int,
    overlap: float = 1.0,
) -> List[Dict[str, Any]]:
    """Process a pair of point clouds and return datapoints.

    Args:
        src_path: Path to the source point cloud file
        tgt_path: Path to the target point cloud file
        transform: Transformation from source to target
        voxel_size: Size of voxel cells for sampling
        min_points: Minimum number of points in a voxel
        max_points: Maximum number of points in a voxel
        overlap: Desired overlap ratio between 0 and 1 (default: 1.0 for full overlap)
    """
    # Load source point cloud
    print(f"Loading source point cloud from {src_path}...")
    src_pc = load_point_cloud(src_path)
    assert isinstance(src_pc, dict)
    assert src_pc.keys() >= {'pos'}
    src_pc = apply_tensor_op(func=lambda x: x.to(transform.device), inputs=src_pc)

    # Load target point cloud
    tgt_pc = load_point_cloud(tgt_path)
    assert isinstance(tgt_pc, dict)
    assert tgt_pc.keys() >= {'pos'}
    tgt_pc = apply_tensor_op(func=lambda x: x.to(transform.device), inputs=tgt_pc)

    # Transform source points to align with target
    transformed_src_pc = copy.deepcopy(src_pc)
    transformed_src_pc['pos'] = apply_transform(src_pc['pos'], transform)

    # Define shifts for partial overlap case
    shift_amount = voxel_size / 2  # Shift by half the voxel size

    # Create a list of target point clouds to process
    # For full overlap (overlap >= 1.0), we only use the original target
    # For partial overlap (overlap < 1.0), we use 6 shifted targets
    shifted_tgt_pcs = []

    if overlap >= 1.0:
        # For full overlap, only use the original target
        shifted_tgt_pcs.append(tgt_pc)
    else:
        # For partial overlap, use 6 shifted targets
        shifts = [
            [shift_amount, 0, 0], [-shift_amount, 0, 0],
            [0, shift_amount, 0], [0, -shift_amount, 0],
            [0, 0, shift_amount], [0, 0, -shift_amount]
        ]

        for shift in shifts:
            # Apply shift to target points
            shifted_tgt_pc = copy.deepcopy(tgt_pc)
            shifted_tgt_pc['pos'][:, 0] += shift[0]
            shifted_tgt_pc['pos'][:, 1] += shift[1]
            shifted_tgt_pc['pos'][:, 2] += shift[2]

            # Add shifted target to the list
            shifted_tgt_pcs.append(shifted_tgt_pc)

    datapoints = []
    # Process each target point cloud
    for shifted_tgt_pc in shifted_tgt_pcs:
        # Apply grid sampling to the union of transformed source and target
        src_voxels, tgt_voxels = grid_sampling([transformed_src_pc, shifted_tgt_pc], voxel_size)
        assert len(src_voxels) == len(tgt_voxels)
        
        # Prepare arguments for parallel processing
        process_args = []
        for src_voxel, tgt_voxel in zip(src_voxels, tgt_voxels):
            process_args.append((
                src_voxel, tgt_voxel, transformed_src_pc, src_pc, tgt_pc,
                src_path, tgt_path, transform, min_points, max_points, overlap, voxel_size
            ))
        
        # Use multiprocessing to process voxel pairs in parallel
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(num_workers) as pool:
            # Use imap_unordered for better performance as order doesn't matter
            # and we want results as soon as they're available
            results = list(pool.imap_unordered(process_voxel_pair, process_args, chunksize=1))
        
        # Filter out None results and add to datapoints
        valid_datapoints = [r for r in results if r is not None]
        datapoints.extend(valid_datapoints)
        
        print(f"Length of datapoints: {len(datapoints)}")

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
        cache_dirname: str = None,
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
        self.cache_dirname = cache_dirname
        super(BasePCRDataset, self).__init__(**kwargs)

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
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), self.cache_dirname+f"_overlap_{self.overlap}")
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")
        self._init_file_pairs()

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

            # Create a partial function with the voxel_size parameter
            process_func = partial(
                process_point_cloud_pair,
                voxel_size=self._voxel_size,
                min_points=self._min_points,
                max_points=self._max_points,
                overlap=self.overlap,
            )

            process_args = []
            for (src_path, tgt_path), transform in zip(self.filepath_pairs, self.transforms):
                process_args.append((src_path, tgt_path, transform))

            # Process files in parallel
            results = [process_func(*args) for args in process_args]

            # Flatten the results list
            self.annotations = [voxel for sublist in results for voxel in sublist]

            # Save voxels to cache in parallel
            print(f"Saving {len(self.annotations)} voxels to cache...")
            save_args = [(i, voxel_data, self.cache_dir) for i, voxel_data in enumerate(self.annotations)]
            with multiprocessing.Pool(num_workers) as pool:
                pool.map(save_datapoint, save_args, chunksize=1)
            print(f"Created and cached {len(self.annotations)} voxels")

        # Split annotations into train/val/test
        self.annotations = self._split_annotations(self.annotations)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint using point indices."""
        # Get voxel data
        annotation = self.annotations[idx]

        # If annotations are filepaths, load the data
        if isinstance(annotation, str):
            annotation = torch.load(annotation)

        # Extract data from voxel_data
        src_points = annotation['src_points']
        tgt_points = annotation['tgt_points']
        src_indices = annotation['src_indices']
        tgt_indices = annotation['tgt_indices']
        src_path = annotation['src_path']
        tgt_path = annotation['tgt_path']
        transform = annotation['transform']

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
