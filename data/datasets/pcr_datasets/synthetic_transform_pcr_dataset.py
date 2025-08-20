from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
import os
import json
import hashlib
import threading
import numpy as np
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap


class SyntheticTransformPCRDataset(BasePCRDataset, ABC):
    """Synthetic transform PCR dataset with transform-to-overlap cache mapping.
    
    Key Concepts:
    - STOCHASTIC phase: Randomly sample transforms, compute overlaps, cache mapping
    - DETERMINISTIC phase: Load from cache for reproducible results
    - Transform-to-overlap mapping: Cache stores transform configs → overlap values
    - Dynamic sizing: Uses annotations from subclass to determine dataset size
    
    Features:
    - Transform-to-overlap cache for reproducible generation
    - Dynamic sizing based on subclass annotations
    - Stochastic sampling → deterministic loading workflow
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
        overlap_range: tuple = (0.3, 1.0),
        min_points: int = 512,
        max_trials: int = 1000,
        cache_filepath: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize synthetic transform PCR dataset.
        
        Args:
            rotation_mag: Maximum rotation magnitude in degrees for synthetic transforms
            translation_mag: Maximum translation magnitude for synthetic transforms
            matching_radius: Radius for correspondence finding
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            min_points: Minimum number of points filter for cache generation
            max_trials: Maximum number of trials to generate valid transforms
            cache_filepath: Path to cache file (if None, no caching is used)
            **kwargs: Additional arguments passed to BaseDataset (including data_root)
        """
        self.overlap_range = tuple(overlap_range)
        self.matching_radius = matching_radius
        self.rotation_mag = rotation_mag
        self.translation_mag = translation_mag
        self.min_points = min_points
        self.max_trials = max_trials
        self.cache_filepath = cache_filepath
        
        # Initialize trials cache (2-level hierarchical)
        if cache_filepath is not None:
            assert cache_filepath.endswith(".json"), f"Cache file must be JSON format, got: {cache_filepath}"
            # Ensure parent directory can be created
            cache_dir = os.path.dirname(cache_filepath)
            if cache_dir:  # Only create if there's actually a directory part
                os.makedirs(cache_dir, exist_ok=True)
            self._load_trials_cache()
        else:
            # No caching
            self.trials_cache = {}
        
        # Initialize cache lock (will be recreated after pickle if needed)
        self._cache_lock = None
        
        super().__init__(**kwargs)
    
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
        """Load cached trials (overlap ratios) from disk."""
        if os.path.exists(self.cache_filepath):
            with open(self.cache_filepath, 'r') as f:
                content = f.read().strip()
                if content:  # Only try to parse if file is not empty
                    loaded_cache = json.loads(content)
                    
                    # Validate cache structure - will raise AssertionError if invalid
                    self._validate_trials_cache_structure(loaded_cache)
                    
                    self.trials_cache = loaded_cache
                else:
                    self.trials_cache = {}
        else:
            self.trials_cache = {}
    
    def _validate_trials_cache_structure(self, cache_data: Any) -> None:
        """Validate that trials cache has the expected 2-level hierarchical structure.
        
        Expected structure:
        {
            "dataset_version_hash": {
                "annotation_hash": [overlap_ratio_1, overlap_ratio_2, ...],
                ...
            },
            ...
        }
        
        Args:
            cache_data: Data loaded from cache file
        """
        # Check if cache_data is a dictionary
        assert isinstance(cache_data, dict), f"Cache data must be a dictionary, got {type(cache_data)}"
        
        # Check each dataset version level (first level)
        for version_key, version_data in cache_data.items():
            # Version key should be a string
            assert isinstance(version_key, str), f"Version key must be string, got {type(version_key)}"
            
            # Version data should be a dictionary
            assert isinstance(version_data, dict), f"Version data must be dictionary, got {type(version_data)}"
            
            # Check each annotation level (second level)
            for annotation_key, overlap_list in version_data.items():
                # Annotation key should be a string
                assert isinstance(annotation_key, str), f"Annotation key must be string, got {type(annotation_key)}"
                
                # Overlap list should be a list of floats
                assert isinstance(overlap_list, list), f"Overlap list must be list, got {type(overlap_list)}"
                
                # Each overlap should be a float
                for i, overlap in enumerate(overlap_list):
                    assert isinstance(overlap, (int, float)), f"Overlap {i} must be number, got {type(overlap)}"
                    assert 0.0 <= overlap <= 1.0, f"Overlap {i} must be in [0, 1], got {overlap}"
    def _save_trials_cache(self) -> None:
        """Save trials cache to disk (thread-safe)."""
        if self.cache_filepath is not None:
            os.makedirs(os.path.dirname(self.cache_filepath), exist_ok=True)
            with open(self.cache_filepath, 'w') as f:
                json.dump(self.trials_cache, f, indent=2)
    
    def _get_dataset_version_key(self) -> str:
        """Generate first-level cache key from dataset version dictionary.
        
        Returns:
            String hash of the dataset version dictionary
        """
        version_dict = self._get_cache_version_dict()
        # Create a deterministic string representation
        version_str = json.dumps(version_dict, sort_keys=True)
        version_hash = hashlib.md5(version_str.encode()).hexdigest()[:12]
        return version_hash
    
    def _get_annotation_cache_key(self, annotation: dict) -> str:
        """Generate second-level cache key from annotation.
        
        Handles torch tensors and other complex objects in annotations by
        serializing them to JSON-compatible format before hashing.
        
        Args:
            annotation: Annotation dictionary from self.annotations[idx]
            
        Returns:
            String hash of the annotation
        """
        # Import serialize_object for handling torch tensors
        from utils.io.json import serialize_object
        
        # Serialize annotation to handle torch tensors and other non-JSON types
        serialized_annotation = serialize_object(annotation)
        
        # Create a deterministic string representation of the serialized annotation
        annotation_str = json.dumps(serialized_annotation, sort_keys=True)
        annotation_hash = hashlib.md5(annotation_str.encode()).hexdigest()[:8]
        return annotation_hash
    
    
    def _load_datapoint(self, idx: int) -> tuple:
        """Load a synthetic datapoint using trial-based caching logic.
        
        Args:
            idx: Dataset index
            
        Returns:
            Tuple of (inputs, labels, meta_info)
        """
        # Assert required fields exist in annotation
        annotation = self.annotations[idx]
        assert 't1_pc_filepath' in annotation, f"annotation[{idx}] missing 't1_pc_filepath', got keys: {list(annotation.keys())}"
        assert 't2_pc_filepath' in annotation, f"annotation[{idx}] missing 't2_pc_filepath', got keys: {list(annotation.keys())}"
        
        # Assert subclass implements _apply_crop
        assert hasattr(self, '_apply_crop'), f"Subclass {self.__class__.__name__} must implement _apply_crop method"
        
        # Extract annotation data
        t1_pc_filepath = annotation['t1_pc_filepath']
        t2_pc_filepath = annotation['t2_pc_filepath']
        
        # Core logic: search for valid cached transform or generate new ones
        src_pc, tgt_pc, overlap_ratio, transform_params, trial_used = self._search_or_generate(
            t1_pc_filepath=t1_pc_filepath,
            t2_pc_filepath=t2_pc_filepath,
            idx=idx
        )
        
        # Build transform matrix for correspondences and output
        transform_matrix = self._build_transform(transform_params)
        
        # Find correspondences
        from utils.point_cloud_ops.correspondences import get_correspondences
        correspondences = get_correspondences(
            src_points=src_pc['pos'],
            tgt_points=tgt_pc['pos'], 
            transform=transform_matrix,
            radius=self.matching_radius,
        )
        
        # Add default features if not present
        if 'feat' not in src_pc:
            src_pc['feat'] = torch.ones((src_pc['pos'].shape[0], 1), dtype=torch.float32, device=src_pc['pos'].device)
        
        if 'feat' not in tgt_pc:
            tgt_pc['feat'] = torch.ones((tgt_pc['pos'].shape[0], 1), dtype=torch.float32, device=tgt_pc['pos'].device)

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
            'trial_idx': trial_used,
            'transform_params': transform_params,
            'overlap': overlap_ratio,
        }
        
        return inputs, labels, meta_info
    
    def _generate(
        self,
        t1_pc_filepath: str,
        t2_pc_filepath: str,
        transform_params: dict,
        idx: int
    ) -> tuple:
        """Generate processed point clouds and overlap ratio for given parameters.
        
        Args:
            t1_pc_filepath: Path to first point cloud file
            t2_pc_filepath: Path to second point cloud file
            transform_params: Transform parameters
            idx: Dataset index for annotation access
            
        Returns:
            Tuple of (src_pc, tgt_pc, overlap_ratio)
        """
        # Load the two point clouds
        from utils.io.point_clouds.load_point_cloud import load_point_cloud
        t1_pc_data = load_point_cloud(t1_pc_filepath, device=self.device, dtype=torch.float64)
        t2_pc_data = load_point_cloud(t2_pc_filepath, device=self.device, dtype=torch.float64)
        
        # Build transform
        transform_matrix = self._build_transform(transform_params)
        
        # Apply inverse transform to PC1 and keep PC2 original
        src_pc_transformed, tgt_pc_original = self._apply_transform(
            t1_pc_data, t2_pc_data, transform_matrix
        )
        
        # Apply crop to both point clouds (build and apply in one step)
        src_pc = self._apply_crop(idx, src_pc_transformed)
        tgt_pc = self._apply_crop(idx, tgt_pc_original)
        
        # Compute overlap ratio
        from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap
        overlap = compute_registration_overlap(
            ref_points=tgt_pc['pos'],
            src_points=src_pc['pos'], 
            transform=transform_matrix,
            positive_radius=self.matching_radius * 2
        )
        
        return src_pc, tgt_pc, float(overlap)
    
    def _search_or_generate(
        self, 
        t1_pc_filepath: str,
        t2_pc_filepath: str,
        idx: int
    ) -> tuple:
        """Search for valid cached transform or generate new ones using trial-based caching.
        
        This is the core logic that handles complete datapoint generation.
        
        Args:
            t1_pc_filepath: Path to first point cloud file
            t2_pc_filepath: Path to second point cloud file
            idx: Dataset index for annotation access
            
        Returns:
            Tuple of (src_pc, tgt_pc, overlap_ratio, transform_params, current_trial)
        """
        # Get cached trial mappings using 2-level hierarchy (thread-safe)
        dataset_version_key = self._get_dataset_version_key()
        annotation = self.annotations[idx]
        annotation_key = self._get_annotation_cache_key(annotation)
        
        # Thread-safe cache structure access
        with self.cache_lock:
            # Get or create cache structure
            if dataset_version_key not in self.trials_cache:
                self.trials_cache[dataset_version_key] = {}
            
            version_cache = self.trials_cache[dataset_version_key]
            if annotation_key not in version_cache:
                version_cache[annotation_key] = []
            
            # Make a copy to avoid concurrent modification issues
            overlap_ratios = version_cache[annotation_key].copy()
        
        # Search through existing cache starting from trial 0
        # trials and datapoint idx are different - trials always start from 0
        current_trial = 0
        while current_trial < self.max_trials:
            # Check if we have this trial cached
            if current_trial < len(overlap_ratios):
                overlap_ratio = overlap_ratios[current_trial]
                # Check if overlap is in valid range
                if self.overlap_range[0] < overlap_ratio <= self.overlap_range[1]:
                    # Found valid trial - generate point clouds with this trial
                    trial_seed = hash((annotation_key, current_trial)) % (2**32)
                    transform_params = self._sample_transform(trial_seed, 0)
                    
                    src_pc, tgt_pc, overlap = self._generate(
                        t1_pc_filepath=t1_pc_filepath,
                        t2_pc_filepath=t2_pc_filepath,
                        transform_params=transform_params,
                        idx=idx
                    )
                    
                    return src_pc, tgt_pc, overlap, transform_params, current_trial
            else:
                # Trial not cached - generate it
                trial_seed = hash((annotation_key, current_trial)) % (2**32)
                transform_params = self._sample_transform(trial_seed, 0)
                
                # Generate point clouds and compute overlap
                src_pc, tgt_pc, overlap_ratio = self._generate(
                    t1_pc_filepath=t1_pc_filepath,
                    t2_pc_filepath=t2_pc_filepath,
                    transform_params=transform_params,
                    idx=idx
                )
                
                # Update cache with thread safety
                with self.cache_lock:
                    # Append to the actual cache list
                    self.trials_cache[dataset_version_key][annotation_key].append(overlap_ratio)
                    if self.cache_filepath is not None:
                        self._save_trials_cache()
                
                # Check if this trial is valid
                if self.overlap_range[0] < overlap_ratio <= self.overlap_range[1]:
                    return src_pc, tgt_pc, overlap_ratio, transform_params, current_trial
            
            current_trial += 1
        
        # If we reach here, no valid transform found within max_trials
        raise RuntimeError(
            f"Failed to find valid transform after {self.max_trials} trials. "
            f"Overlap range: {self.overlap_range}, annotation_key: {annotation_key}"
        )
    
    def _sample_transform(self, seed: int, file_idx: int) -> dict:
        """Sample SE(3) transformation parameters only.
        
        Standard implementation that works for all subclasses.
        
        Args:
            seed: Random seed for deterministic sampling
            file_idx: Index of file pair (unused but kept for API consistency)
            
        Returns:
            Dictionary containing SE(3) transform parameters:
            - 'rotation_angles': List of 3 rotation angles in degrees
            - 'translation': List of 3 translation values
        """
        _ = file_idx  # Suppress unused parameter warning
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Sample SE(3) transformation parameters
        rotation_angles = torch.rand(3, generator=generator) * (2 * self.rotation_mag) - self.rotation_mag
        translation = torch.rand(3, generator=generator) * (2 * self.translation_mag) - self.translation_mag
        
        return {
            'rotation_angles': rotation_angles.tolist(),
            'translation': translation.tolist(),
        }
    
    def _build_transform(self, transform_params: dict) -> torch.Tensor:
        """Build SE(3) transformation matrix only.
        
        Standard implementation that works for all subclasses.
        
        Args:
            transform_params: Transform configuration dictionary from _sample_transform
            
        Returns:
            4x4 SE(3) transformation matrix (torch.Tensor)
        """
        # Convert to tensors
        rotation_angles = torch.tensor(transform_params['rotation_angles'], dtype=torch.float32, device=self.device)
        translation = torch.tensor(transform_params['translation'], dtype=torch.float32, device=self.device)
        
        # Convert to radians and create rotation matrices
        rotation_rad = rotation_angles * np.pi / 180
        cos_vals = torch.cos(rotation_rad)
        sin_vals = torch.sin(rotation_rad)
        
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_vals[0], -sin_vals[0]],
            [0, sin_vals[0], cos_vals[0]]
        ], dtype=torch.float32, device=self.device)
        
        R_y = torch.tensor([
            [cos_vals[1], 0, sin_vals[1]],
            [0, 1, 0],
            [-sin_vals[1], 0, cos_vals[1]]
        ], dtype=torch.float32, device=self.device)
        
        R_z = torch.tensor([
            [cos_vals[2], -sin_vals[2], 0],
            [sin_vals[2], cos_vals[2], 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        rotation_matrix = R_z @ R_y @ R_x
        
        # Create 4x4 transformation matrix
        transform_matrix = torch.eye(4, dtype=torch.float32, device=self.device)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        return transform_matrix
    
    def _apply_transform(self, src_pc_data: dict, tgt_pc_data: dict, 
                        transform_matrix: torch.Tensor) -> tuple:
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
        from utils.point_cloud_ops import apply_transform
        # Create a fresh tensor to avoid threading issues
        transform_inv = torch.linalg.inv(transform_matrix.detach().cpu()).to(transform_matrix.device)
        src_points = apply_transform(src_pc_data['pos'], transform_inv)
        
        # Update point cloud dictionaries with transformed/original positions
        src_pc_data_transformed = src_pc_data.copy()
        src_pc_data_transformed['pos'] = src_points
        
        tgt_pc_data_original = tgt_pc_data.copy()
        tgt_pc_data_original['pos'] = tgt_points
        
        return src_pc_data_transformed, tgt_pc_data_original
    
    
    @abstractmethod
    def _apply_crop(self, idx: int, pc_data: dict) -> dict:
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
