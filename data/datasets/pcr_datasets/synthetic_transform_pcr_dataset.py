import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from utils.determinism.hash_utils import deterministic_hash
from utils.io.json import save_json
from utils.io.point_clouds.load_point_cloud import load_point_cloud
from utils.point_cloud_ops import apply_transform
from utils.point_cloud_ops.correspondences import get_correspondences
from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap


class SyntheticTransformPCRDataset(BasePCRDataset, ABC):
    """Synthetic transform PCR dataset with transform-to-overlap cache mapping.

    Key Concepts:
    - STOCHASTIC phase: Randomly sample transforms, compute overlaps, cache mapping
    - DETERMINISTIC phase: Load from cache for reproducible results
    - Transform-to-overlap mapping: Cache stores transform configs -> overlap values
    - Dynamic sizing: Uses annotations from subclass to determine dataset size

    Features:
    - Transform-to-overlap cache for reproducible generation
    - Dynamic sizing based on subclass annotations
    - Stochastic sampling -> deterministic loading workflow
    - No try-catch blocks - fail fast with clear errors
    """

    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None
    SHA1SUM = None

    def __init__(
        self,
        rotation_mag: float = 45.0,
        translation_mag: float = 0.5,
        matching_radius: float = 0.05,
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        min_points: int = 512,
        max_trials: int = 1000,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize synthetic transform PCR dataset.

        Args:
            rotation_mag: Maximum rotation magnitude in degrees for synthetic transforms
            translation_mag: Maximum translation magnitude for synthetic transforms
            matching_radius: Radius for correspondence finding
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            min_points: Minimum number of points filter for cache generation
            max_trials: Maximum number of trials to generate valid transforms
            cache_dir: Directory to store cache files (if None, no caching is used)
            **kwargs: Additional arguments passed to BaseDataset (including data_root)
        """
        self.overlap_range = overlap_range
        self.matching_radius = matching_radius
        self.rotation_mag = rotation_mag
        self.translation_mag = translation_mag
        self.min_points = min_points
        self.max_trials = max_trials
        self.cache_dir = cache_dir

        # Initialize cache lock (will be recreated after pickle if needed)
        self._cache_lock = None

        super().__init__(**kwargs)

        # Initialize trials cache after super().__init__ so we have access to get_cache_version_hash()
        if cache_dir is not None:
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            # Compute and store cache filepath once
            version_hash = self.get_cache_version_hash()
            self.cache_filepath = os.path.join(cache_dir, f"{version_hash}.json")
            self._load_trials_cache()
        else:
            # No caching
            self.cache_filepath = None
            self.trials_cache = {}

    @property
    def cache_lock(self) -> threading.Lock:
        """Lazy initialization of cache lock to handle pickle/unpickle scenarios."""
        if self._cache_lock is None:
            self._cache_lock = threading.Lock()
        return self._cache_lock

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'rotation_mag': self.rotation_mag,
            'translation_mag': self.translation_mag,
            'matching_radius': self.matching_radius,
            'overlap_range': self.overlap_range,
            'min_points': self.min_points,
            'max_trials': self.max_trials,
        })
        return version_dict

    def _init_annotations(self) -> None:
        """Initialize annotations for SyntheticTransformPCRDataset.

        This method should be overridden by subclasses that have their own
        annotation structure (e.g., bi-temporal datasets).

        For standalone synthetic datasets, this method should create annotations
        based on available data.
        """
        # This will be overridden by subclasses
        # For example, iVISION_PCR_Dataset will use bi-temporal annotations
        raise NotImplementedError("Subclasses must implement _init_annotations")

    def _load_trials_cache(self) -> None:
        """Load cached trials (overlap ratios) from version-specific cache file."""
        assert self.cache_filepath is not None
        if os.path.exists(self.cache_filepath):
            with open(self.cache_filepath, 'r') as f:
                loaded_cache = json.load(f)

                # Validate cache structure - will raise AssertionError if invalid
                self._validate_trials_cache_structure(loaded_cache)

                self.trials_cache = loaded_cache
        else:
            # Create empty cache file for this version
            self.trials_cache = {}
            with self.cache_lock:
                self._save_trials_cache()

    def _validate_trials_cache_structure(self, cache_data: Any) -> None:
        """Validate that trials cache has the expected structure.

        Expected structure:
        {
            "0": [overlap_ratio_1, overlap_ratio_2, ...],
            "1": [overlap_ratio_1, overlap_ratio_2, ...],
            ...
        }

        Args:
            cache_data: Data loaded from cache file
        """
        # Check if cache_data is a dictionary
        assert isinstance(cache_data, dict), (
            f"Cache data must be a dictionary, got {type(cache_data)}"
        )

        # Check each dataset index level
        for idx_key, overlap_list in cache_data.items():
            # Index key should be a string (representing dataset index)
            assert isinstance(idx_key, str), (
                f"Index key must be string, got {type(idx_key)}"
            )

            # Check if key can be converted to int (valid dataset index)
            try:
                int(idx_key)
            except ValueError:
                assert False, f"Index key must be convertible to int, got: {idx_key}"

            # Overlap list should be a list
            assert isinstance(overlap_list, list), (
                f"Overlap list must be list, got {type(overlap_list)}"
            )

            # Each overlap should be a float or None (for failed generations)
            for i, overlap in enumerate(overlap_list):
                if overlap is not None:
                    assert isinstance(overlap, (int, float)), (
                        f"Overlap {i} for idx {idx_key} must be number or None, got {type(overlap)}"
                    )
                    assert 0.0 <= overlap <= 1.0, (
                        f"Overlap {i} for idx {idx_key} must be in [0, 1], got {overlap}"
                    )

    def _save_trials_cache(self) -> None:
        """Save trials cache to version-specific cache file (thread-safe)."""
        assert self.cache_filepath is not None
        save_json(self.trials_cache, self.cache_filepath)

    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Load a synthetic datapoint using trial-based caching logic.

        Args:
            idx: Dataset index

        Returns:
            Tuple of (inputs, labels, meta_info)
        """
        # Assert required fields exist in annotation
        annotation = self.annotations[idx]
        assert 't1_pc_filepath' in annotation, (
            f"annotation[{idx}] missing 't1_pc_filepath', got keys: {list(annotation.keys())}"
        )
        assert 't2_pc_filepath' in annotation, (
            f"annotation[{idx}] missing 't2_pc_filepath', got keys: {list(annotation.keys())}"
        )

        # Assert subclass implements _apply_crop
        assert hasattr(self, '_apply_crop'), (
            f"Subclass {self.__class__.__name__} must implement _apply_crop method"
        )

        # Extract annotation data
        t1_pc_filepath = annotation['t1_pc_filepath']
        t2_pc_filepath = annotation['t2_pc_filepath']

        # Core logic: search for valid cached transform or generate new ones
        src_pc, tgt_pc, overlap_ratio, transform_matrix, trial_idx = self._search_or_generate(
            t1_pc_filepath=t1_pc_filepath,
            t2_pc_filepath=t2_pc_filepath,
            idx=idx,
        )

        # Find correspondences
        correspondences = get_correspondences(
            src_points=src_pc['pos'],
            tgt_points=tgt_pc['pos'],
            transform=transform_matrix,
            radius=self.matching_radius,
        )

        # Add default features if not present
        if 'feat' not in src_pc:
            src_pc['feat'] = torch.ones(
                (src_pc['pos'].shape[0], 1),
                dtype=torch.float32,
                device=src_pc['pos'].device,
            )

        if 'feat' not in tgt_pc:
            tgt_pc['feat'] = torch.ones(
                (tgt_pc['pos'].shape[0], 1),
                dtype=torch.float32,
                device=tgt_pc['pos'].device,
            )

        inputs = {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': correspondences,
            'transform': transform_matrix,
        }

        labels = {
            'transform': transform_matrix,
        }

        meta_info = {
            't1_pc_filepath': t1_pc_filepath,
            't2_pc_filepath': t2_pc_filepath,
            'trial_idx': trial_idx,
            'transform_matrix': transform_matrix,
            'overlap': overlap_ratio,
        }

        return inputs, labels, meta_info

    def _generate(
        self,
        t1_pc_filepath: str,
        t2_pc_filepath: str,
        transform_matrix: torch.Tensor,
        idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[float]]:
        """Generate processed point clouds and overlap ratio for given parameters.

        Args:
            t1_pc_filepath: Path to first point cloud file
            t2_pc_filepath: Path to second point cloud file
            transform_matrix: 4x4 transformation matrix
            idx: Dataset index for annotation access

        Returns:
            Tuple of (src_pc, tgt_pc, overlap_ratio)
        """
        # Load the two point clouds
        t1_pc_data = load_point_cloud(
            t1_pc_filepath,
            device=self.device,
            dtype=torch.float64,
        )
        t2_pc_data = load_point_cloud(
            t2_pc_filepath,
            device=self.device,
            dtype=torch.float64,
        )

        # Apply inverse transform to PC1 and keep PC2 original
        src_pc_transformed, tgt_pc_original = self._apply_transform(
            t1_pc_data,
            t2_pc_data,
            transform_matrix,
        )

        # Apply crop to both point clouds (build and apply in one step)
        src_pc = self._apply_crop(idx, src_pc_transformed)
        tgt_pc = self._apply_crop(idx, tgt_pc_original)

        # Check if crops resulted in empty point clouds
        if src_pc['pos'].shape[0] == 0 or tgt_pc['pos'].shape[0] == 0:
            overlap_ratio = None
        else:
            overlap_ratio = compute_registration_overlap(
                ref_points=tgt_pc['pos'],
                src_points=src_pc['pos'],
                transform=transform_matrix,
                positive_radius=self.matching_radius * 2,
            )

        return src_pc, tgt_pc, overlap_ratio

    def _search_or_generate(
        self,
        t1_pc_filepath: str,
        t2_pc_filepath: str,
        idx: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, torch.Tensor, int]:
        """Search for valid cached transform or generate new ones using trial-based caching.

        This method iterates through trials (starting from 0) to find a transform that
        produces overlap within the specified range. It uses dataset indices as cache keys
        to store overlap ratios for each trial.

        Args:
            t1_pc_filepath: Path to first point cloud file
            t2_pc_filepath: Path to second point cloud file
            idx: Dataset index for annotation access

        Returns:
            Tuple of (src_pc, tgt_pc, overlap_ratio, transform_matrix, current_trial)
        """
        # Use dataset index as cache key
        idx_key = str(idx)

        # Thread-safe cache structure initialization
        with self.cache_lock:
            if idx_key not in self.trials_cache:
                self.trials_cache[idx_key] = []

            # Make a copy to avoid concurrent modification issues
            cached_overlaps = self.trials_cache[idx_key].copy()

        # Iterate through trials to find valid transform
        for trial_idx in range(self.max_trials):
            # Generate transform matrix for this trial
            trial_seed = deterministic_hash((idx, trial_idx))
            transform_matrix = self._sample_transform(trial_seed)

            # Process cached trial
            if trial_idx < len(cached_overlaps):
                cached_overlap = cached_overlaps[trial_idx]

                # Check if cached overlap is valid and in range
                if cached_overlap is not None and self.overlap_range[0] < cached_overlap <= self.overlap_range[1]:
                    # Valid cached trial - generate point clouds
                    src_pc, tgt_pc, generated_overlap = self._generate(
                        t1_pc_filepath=t1_pc_filepath,
                        t2_pc_filepath=t2_pc_filepath,
                        transform_matrix=transform_matrix,
                        idx=idx,
                    )

                    assert generated_overlap is not None, (
                        f"Generated overlap should not be None for cached valid trial {trial_idx}"
                    )
                    assert abs(generated_overlap - cached_overlap) < 1e-5, (
                        f"Generated overlap {generated_overlap} should match cached value {cached_overlap} "
                        f"for trial {trial_idx} (diff: {abs(generated_overlap - cached_overlap)})"
                    )

                    print(f"DEBUG: _search_or_generate: datapoint {idx} using cached trial {trial_idx}.")
                    return src_pc, tgt_pc, generated_overlap, transform_matrix, trial_idx

            else:
                # Process new trial (not cached)
                src_pc, tgt_pc, overlap_ratio = self._generate(
                    t1_pc_filepath=t1_pc_filepath,
                    t2_pc_filepath=t2_pc_filepath,
                    transform_matrix=transform_matrix,
                    idx=idx,
                )

                # Cache the result (thread-safe)
                with self.cache_lock:
                    cache_list = self.trials_cache[idx_key]
                    assert len(cache_list) == trial_idx, (
                        f"Expected cache list length {trial_idx}, got {len(cache_list)}"
                    )
                    cache_list.append(overlap_ratio)

                    if self.cache_filepath is not None:
                        self._save_trials_cache()

                # Check if new trial produces valid overlap
                if overlap_ratio is not None and self.overlap_range[0] < overlap_ratio <= self.overlap_range[1]:
                    print(f"DEBUG: _search_or_generate: datapoint {idx} using newly generated trial {trial_idx}...")
                    return src_pc, tgt_pc, overlap_ratio, transform_matrix, trial_idx

        # No valid transform found within max_trials
        raise RuntimeError(
            f"Failed to find valid transform after {self.max_trials} trials. "
            f"Overlap range: {self.overlap_range}, idx: {idx}, annotation: {self.annotations[idx]}"
        )

    def _sample_transform(self, seed: int) -> torch.Tensor:
        """Sample SE(3) transformation matrix directly.

        Standard implementation that works for all subclasses.
        Samples rotation and translation with proper magnitude constraints:
        - rotation_mag: Maximum overall rotation magnitude (degrees)
        - translation_mag: Maximum overall translation magnitude

        Args:
            seed: Random seed for deterministic sampling

        Returns:
            4x4 SE(3) transformation matrix (torch.Tensor)
        """
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Sample rotation using axis-angle representation to ensure proper magnitude constraint
        # Sample random rotation axis (unit vector)
        rotation_axis = torch.randn(3, generator=generator)
        rotation_axis = rotation_axis / torch.norm(rotation_axis)

        # Sample rotation angle uniformly from [0, rotation_mag] (in degrees)
        rotation_angle_deg = torch.rand(1, generator=generator) * self.rotation_mag
        rotation_angle_rad = rotation_angle_deg * np.pi / 180

        # Convert axis-angle to rotation matrix using Rodrigues' formula
        K = torch.tensor([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ], dtype=torch.float32)

        R = (
            torch.eye(3, dtype=torch.float32)
            + torch.sin(rotation_angle_rad) * K
            + (1 - torch.cos(rotation_angle_rad)) * (K @ K)
        )

        # Assert that the rotation matrix corresponds to the sampled angle
        computed_angle = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1))
        assert torch.abs(computed_angle - rotation_angle_rad) < 1e-4, \
            f"Rotation matrix angle mismatch: sampled={rotation_angle_rad.item():.6f}, computed={computed_angle.item():.6f}"

        # Sample translation with proper magnitude constraint
        # Sample random direction on unit sphere
        translation_direction = torch.randn(3, generator=generator)
        translation_direction = translation_direction / torch.norm(translation_direction)

        # Sample magnitude uniformly from [0, translation_mag]
        translation_magnitude = torch.rand(1, generator=generator) * self.translation_mag

        # Apply magnitude to direction vector
        translation = translation_direction * translation_magnitude

        # Create 4x4 transformation matrix
        transform_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        transform_matrix[:3, :3] = R.to(self.device)
        transform_matrix[:3, 3] = translation.to(self.device)

        return transform_matrix

    def _apply_transform(
        self,
        src_pc_data: Dict[str, torch.Tensor],
        tgt_pc_data: Dict[str, torch.Tensor],
        transform_matrix: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply SE(3) transformation only.

        Standard PCR convention: source + gt_transform = target
        Following GeoTransformer's approach:
        1. ref_points (target) = original point cloud
        2. src_points (source) = apply inverse transform to create misaligned source

        Args:
            src_pc_data: Raw source point cloud dictionary (pos, rgb, etc.)
            tgt_pc_data: Raw target point cloud dictionary (same as src_pc_data for single-temporal)
            transform_matrix: 4x4 transformation matrix to align source to target

        Returns:
            Tuple of (src_pc_transformed, tgt_pc_original) dictionaries
        """
        transform_matrix = transform_matrix.to(src_pc_data['pos'].dtype)
        # Following GeoTransformer's approach:
        # ref_points = original (target)
        tgt_points = tgt_pc_data['pos'].clone()

        # src_points = apply inverse transform to create misaligned source
        transform_inv = torch.linalg.inv(
            transform_matrix.detach().cpu()
        ).to(transform_matrix.device)
        src_points = apply_transform(src_pc_data['pos'], transform_inv)

        # Update point cloud dictionaries with transformed/original positions
        src_pc_data_transformed = src_pc_data.copy()
        src_pc_data_transformed['pos'] = src_points

        tgt_pc_data_original = tgt_pc_data.copy()
        tgt_pc_data_original['pos'] = tgt_points

        return src_pc_data_transformed, tgt_pc_data_original

    @abstractmethod
    def _apply_crop(self, idx: int, pc_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build and apply crop transform to point cloud data.

        Subclasses must implement this method to build the crop transform
        from annotation data and apply it to the point cloud.

        Args:
            idx: Index into self.annotations
            pc_data: Point cloud dictionary

        Returns:
            Cropped point cloud dictionary
        """
        raise NotImplementedError("Subclasses must implement _apply_crop")
