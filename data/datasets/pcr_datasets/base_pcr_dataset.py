from typing import Tuple, Dict, Any, List
import copy
import os
import glob
import numpy as np
import torch
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.datasets.base_dataset import BaseDataset
from utils.io import load_point_cloud
from utils.point_cloud_ops.correspondences import get_correspondences
from utils.point_cloud_ops.apply_transform import apply_transform
from utils.point_cloud_ops.grid_sampling import grid_sampling
from utils.point_cloud_ops.set_ops.intersection import compute_pc_iou
from utils.point_cloud_ops.select import Select
from utils.point_cloud_ops.random_select import RandomSelect
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
            radius=1.0,
        )
        # Skip if overlap ratio is not within the desired range (±10%)
        if abs(overlap_ratio - overlap) > 0.1:
            return None
    else:
        overlap_ratio = 1.0

    # Apply random sampling if needed
    src_pc_final = Select(src_voxel['indices'])(src_pc)
    if len(src_voxel['indices']) > max_points:
        src_pc_final = RandomSelect(percentage=max_points / len(src_voxel['indices']))(src_pc_final)

    tgt_pc_final = Select(tgt_voxel['indices'])(tgt_pc)
    if len(tgt_voxel['indices']) > max_points:
        tgt_pc_final = RandomSelect(percentage=max_points / len(tgt_voxel['indices']))(tgt_pc_final)

    # Create datapoint with position and RGB if available
    datapoint = {
        'src_points': src_pc_final['pos'],
        'src_indices': src_pc_final['indices'],
        'src_path': src_path,
        'tgt_points': tgt_pc_final['pos'],
        'tgt_indices': tgt_pc_final['indices'],
        'tgt_path': tgt_path,
        'transform': transform,
        'overlap_ratio': overlap_ratio,
    }

    # Add RGB colors if available
    if 'rgb' in src_pc_final:
        datapoint['src_rgb'] = src_pc_final['rgb']
    if 'rgb' in tgt_pc_final:
        datapoint['tgt_rgb'] = tgt_pc_final['rgb']

    return datapoint


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

    # Define base directions for shifts
    SHIFT_DIRECTIONS = [
        # Principal axes
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],

        # Diagonal planes
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],

        # 3D diagonals
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
        [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
    ]

    # Define magnitudes for shifts
    SHIFT_MAGNITUDES = [0.75, 1.0, 1.25]

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

    @property
    def shifts(self):
        """Generate a list of shifts based on directions and magnitudes.

        Returns:
            List of shift vectors
        """
        import itertools

        # Calculate shift amount based on voxel size and overlap
        shift_amount = self._voxel_size * (1 - self.overlap)

        shifts = []
        for direction, magnitude in itertools.product(self.SHIFT_DIRECTIONS, self.SHIFT_MAGNITUDES):
            # Calculate the magnitude of the direction vector
            dir_magnitude = sum(x*x for x in direction) ** 0.5

            # Normalize and apply magnitude and shift_amount in one step
            shift = [d / dir_magnitude * magnitude * shift_amount for d in direction]
            shifts.append(shift)

        return shifts

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.

        This method should be overridden by subclasses to set up:
        - self.src_file_paths: List of source file paths
        - self.tgt_file_paths: List of target file paths
        - self.gt_transforms: List of transforms from source to target
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
        start_time = time.time()

        # Get file paths
        self.file_paths = sorted(glob.glob(os.path.join(self.data_root, '*.las')))
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), self.cache_dirname+f"_overlap_{self.overlap}")
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Found {len(self.file_paths)} point clouds in {self.data_root}.")
        self._init_file_pairs()

        # Check if cache exists
        scene_dirs = [
            os.path.join(self.cache_dir, f'scene_pair_{idx}')
            for idx in range(len(self.filepath_pairs))
            if glob.glob(os.path.join(self.cache_dir, f'scene_pair_{idx}', 'voxel_*.pt'))
        ]
        if False and len(scene_dirs) > 0:
            # Load all voxel files from all scene directories
            self.annotations = []
            for scene_dir in scene_dirs:
                voxel_files = sorted(glob.glob(os.path.join(scene_dir, 'voxel_*.pt')))
                self.annotations.extend(voxel_files)
            elapsed = time.time() - start_time
            print(f"Loaded {len(self.annotations)} cached voxels from {len(scene_dirs)} scene pairs in {elapsed:.2f} seconds")
        else:
            # Process point clouds
            print("Processing point clouds...")

            self.annotations = []
            total_pairs = len(self.filepath_pairs)
            for pair_idx, ((src_path, tgt_path), transform) in enumerate(zip(self.filepath_pairs, self.gt_transforms)):
                if src_path == '/home/daniel/repos/Pylon/./data/datasets/soft_links/ivision-pcr-data/Week-03-Wed-Oct-09-2024.las':
                    continue
                pair_start_time = time.time()
                # Create a directory for this scene pair
                scene_dir = os.path.join(self.cache_dir, f'scene_pair_{pair_idx}')
                os.makedirs(scene_dir, exist_ok=True)

                # Check if this scene pair has already been processed
                existing_voxels = sorted(glob.glob(os.path.join(scene_dir, 'voxel_*.pt')))
                if len(existing_voxels) > 0:
                    print(f"Scene pair {pair_idx}/{total_pairs-1} already processed, loading {len(existing_voxels)} voxels")
                    self.annotations.extend(existing_voxels)
                    continue

                print(f"Processing scene pair {pair_idx}/{total_pairs-1}...")
                # Process the point cloud pair and get the file paths of saved voxels
                voxel_file_paths = self._process_point_cloud_pair(
                    src_path, tgt_path, transform,
                    scene_dir
                )
                # Add the file paths to annotations
                self.annotations.extend(voxel_file_paths)

                pair_elapsed = time.time() - pair_start_time
                print(f"Completed scene pair {pair_idx}/{total_pairs-1} in {pair_elapsed:.2f} seconds")

            total_elapsed = time.time() - start_time
            print(f"Created and cached {len(self.annotations)} voxels in {total_elapsed:.2f} seconds")

        # Split annotations into train/val/test
        self.annotations = self._split_annotations(self.annotations)
        print(f"Dataset initialization completed in {time.time() - start_time:.2f} seconds")

    def _process_point_cloud_pair(
        self, src_path: str, tgt_path: str, transform: torch.Tensor,
        scene_dir: str = None,
    ) -> List[str]:
        """Process a pair of point clouds and return file paths to saved datapoints.

        Args:
            src_path: Path to the source point cloud file
            tgt_path: Path to the target point cloud file
            transform: Transformation from source to target
            scene_dir: Directory to save datapoints for this scene pair

        Returns:
            List of file paths to saved datapoints
        """
        pair_start_time = time.time()

        # Load source point cloud
        print(f"Loading source point cloud from {src_path}...")
        src_load_start = time.time()
        src_pc = load_point_cloud(src_path)
        assert isinstance(src_pc, dict)
        assert src_pc.keys() >= {'pos'}
        src_pc = apply_tensor_op(func=lambda x: x.to(self.device), inputs=src_pc)
        print(f"Source point cloud loaded in {time.time() - src_load_start:.2f} seconds")

        # Load target point cloud
        print(f"Loading target point cloud from {tgt_path}...")
        tgt_load_start = time.time()
        tgt_pc = load_point_cloud(tgt_path)
        assert isinstance(tgt_pc, dict)
        assert tgt_pc.keys() >= {'pos'}
        tgt_pc = apply_tensor_op(func=lambda x: x.to(self.device), inputs=tgt_pc)
        print(f"Target point cloud loaded in {time.time() - tgt_load_start:.2f} seconds")

        # Move transform to device
        transform = transform.to(self.device)

        # Transform source points to align with target
        transformed_src_pc = copy.deepcopy(src_pc)
        transformed_src_pc['pos'] = apply_transform(src_pc['pos'], transform)

        datapoint_file_paths = []
        enumeration_start = 0

        # Process target point clouds based on overlap setting
        if self.overlap == 1.0:
            # For full overlap, only use the original target
            print("Using full overlap (overlap >= 1.0), processing only the original target...")

            # Process the original target
            file_paths = self._process_target_point_cloud(
                transformed_src_pc=transformed_src_pc,
                target_pc=tgt_pc,
                src_pc=src_pc,
                tgt_pc=tgt_pc,
                src_path=src_path,
                tgt_path=tgt_path,
                transform=transform,
                scene_dir=scene_dir,
                enumeration_start=enumeration_start
            )

            datapoint_file_paths.extend(file_paths)
            print(f"Total datapoint files: {len(datapoint_file_paths)}")

        else:
            # For partial overlap, use only the shifted targets
            print(f"Using partial overlap (overlap = {self.overlap}), processing {len(self.shifts)} shifted targets...")

            # Process each shifted target
            for shift_idx, shift in enumerate(self.shifts, 1):
                shift_start_time = time.time()
                print(f"Processing shifted target {shift_idx}/{len(self.shifts)}...")

                # Apply shift to target points
                shifted_tgt_pc = copy.deepcopy(tgt_pc)
                shifted_tgt_pc['pos'][:, 0] += shift[0]
                shifted_tgt_pc['pos'][:, 1] += shift[1]
                shifted_tgt_pc['pos'][:, 2] += shift[2]

                # Process the shifted target with updated enumeration_start
                file_paths = self._process_target_point_cloud(
                    transformed_src_pc=transformed_src_pc,
                    target_pc=shifted_tgt_pc,
                    src_pc=src_pc,
                    tgt_pc=tgt_pc,
                    src_path=src_path,
                    tgt_path=tgt_path,
                    transform=transform,
                    scene_dir=scene_dir,
                    enumeration_start=enumeration_start
                )

                datapoint_file_paths.extend(file_paths)
                # Update enumeration_start for the next shift
                enumeration_start += len(file_paths)
                print(f"Shifted target {shift_idx}/{len(self.shifts)} completed in {time.time() - shift_start_time:.2f} seconds")
                print(f"Total datapoint files so far: {len(datapoint_file_paths)}")

        total_elapsed = time.time() - pair_start_time
        print(f"Scene pair processing completed in {total_elapsed:.2f} seconds with {len(datapoint_file_paths)} total datapoint files")
        return datapoint_file_paths

    def _process_target_point_cloud(
        self, transformed_src_pc, target_pc, src_pc, tgt_pc,
        src_path, tgt_path, transform, scene_dir, enumeration_start=0
    ) -> List[str]:
        """Process a single target point cloud and return file paths to saved datapoints.

        Args:
            transformed_src_pc: Transformed source point cloud
            target_pc: Target point cloud to process
            src_pc: Original source point cloud
            tgt_pc: Original target point cloud
            src_path: Path to the source point cloud file
            tgt_path: Path to the target point cloud file
            transform: Transformation from source to target
            scene_dir: Directory to save datapoints
            enumeration_start: Starting index for file enumeration to avoid overwriting

        Returns:
            List of file paths to saved datapoints
        """
        num_workers = max(1, multiprocessing.cpu_count() - 1)

        # Apply grid sampling to the union of transformed source and target
        print(f"Grid sampling using {num_workers} workers...")
        grid_start_time = time.time()
        src_voxels, tgt_voxels = grid_sampling([transformed_src_pc, target_pc], self._voxel_size, num_workers=num_workers)
        assert len(src_voxels) == len(tgt_voxels)
        print(f"Grid sampling completed in {time.time() - grid_start_time:.2f} seconds")

        # Process voxel pairs in parallel
        print(f"Processing {len(src_voxels)} voxel pairs using {num_workers} workers...")
        process_start_time = time.time()
        process_args = []
        for src_voxel, tgt_voxel in zip(src_voxels, tgt_voxels):
            process_args.append((
                src_voxel, tgt_voxel, transformed_src_pc, src_pc, tgt_pc,
                src_path, tgt_path, transform, self._min_points, self._max_points, self.overlap, self._voxel_size
            ))

        # Use ProcessPoolExecutor instead of Pool
        valid_datapoints = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_voxel_pair, args): args for args in process_args}

            # Process results as they complete
            for future in as_completed(future_to_args):
                # This will raise any exceptions that occurred in the worker process
                result = future.result()
                if result is not None:
                    valid_datapoints.append(result)

        print(f"Voxel pair processing completed in {time.time() - process_start_time:.2f} seconds")

        # Save datapoint cache files in parallel and collect file paths
        print(f"Saving {len(valid_datapoints)} voxels to {scene_dir} using {num_workers} workers...")
        save_start_time = time.time()

        # Create file paths for each datapoint with enumeration_start offset
        file_paths = [os.path.join(scene_dir, f'voxel_{i + enumeration_start}.pt') for i in range(len(valid_datapoints))]

        # Save datapoints in parallel
        save_args = [(datapoint, file_path) for datapoint, file_path in zip(valid_datapoints, file_paths)]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(torch.save, datapoint, file_path): (datapoint, file_path)
                             for datapoint, file_path in save_args}

            # Process results as they complete - this will raise any exceptions
            for future in as_completed(future_to_args):
                # This will raise any exceptions that occurred in the worker process
                future.result()

        print(f"Saved {len(valid_datapoints)} voxels to {scene_dir} in {time.time() - save_start_time:.2f} seconds")

        return file_paths

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        """Load a datapoint using point indices.

        The annotations are file paths to cached voxel files, organized by scene pairs.
        Each scene pair has its own directory (scene_pair_X) with voxel files (voxel_Y.pt).
        """
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

        # Initialize inputs with position and feature
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

        # Add RGB colors if available
        if 'src_rgb' in annotation:
            inputs['src_pc']['rgb'] = annotation['src_rgb']
        if 'tgt_rgb' in annotation:
            inputs['tgt_pc']['rgb'] = annotation['tgt_rgb']

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
