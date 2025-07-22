from typing import Tuple, Dict, Any, List, Optional, Union
from abc import ABC
import os
import json
import hashlib
import threading
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap
from data.transforms.vision_3d.lidar_simulation_crop import LiDARSimulationCrop
from utils.io.point_cloud import load_point_cloud


class SyntheticTransformPCRDataset(BaseDataset, ABC):
    """Synthetic transform PCR dataset with transform-to-overlap cache mapping.
    
    Key Concepts:
    - STOCHASTIC phase: Randomly sample transforms, compute overlaps, cache mapping
    - DETERMINISTIC phase: Load from cache for reproducible results
    - Transform-to-overlap mapping: Cache stores transform configs → overlap values
    - Even distribution: Divide dataset_size evenly across source files/pairs
    
    Features:
    - Transform-to-overlap cache for reproducible generation
    - Even distribution of synthetic pairs across source data
    - Stochastic sampling → deterministic loading workflow
    - No try-catch blocks - fail fast with clear errors
    """
    
    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']  
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        data_root: str,
        dataset_size: int,
        rotation_mag: float = 45.0,
        translation_mag: float = 0.5,
        matching_radius: float = 0.05,
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        min_points: int = 512,
        max_trials: int = 1000,
        cache_filepath: Optional[str] = None,
        crop_method: str = 'lidar',
        lidar_max_range: float = 6.0,
        lidar_horizontal_fov: Union[int, float] = 120.0,
        lidar_vertical_fov: Union[int, float] = 60.0,
        lidar_apply_range_filter: bool = False,
        lidar_apply_fov_filter: bool = True,
        lidar_apply_occlusion_filter: bool = False,
        **kwargs,
    ) -> None:
        """Initialize synthetic transform PCR dataset.
        
        Args:
            data_root: Path to dataset root directory
            dataset_size: Total number of synthetic pairs to generate
            rotation_mag: Maximum rotation magnitude in degrees for synthetic transforms
            translation_mag: Maximum translation magnitude for synthetic transforms
            matching_radius: Radius for correspondence finding
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            min_points: Minimum number of points filter for cache generation
            max_trials: Maximum number of trials to generate valid transforms
            cache_filepath: Path to cache file (if None, no caching is used)
            crop_method: Cropping method - only 'lidar' is supported
            lidar_max_range: Maximum LiDAR sensor range in meters
            lidar_horizontal_fov: LiDAR horizontal field of view in degrees
            lidar_vertical_fov: LiDAR vertical FOV total angle in degrees (e.g., 60.0 means [-30.0, +30.0])
            lidar_apply_range_filter: Whether to apply range-based filtering for LiDAR (default: False)
            lidar_apply_fov_filter: Whether to apply field-of-view filtering for LiDAR (default: True)  
            lidar_apply_occlusion_filter: Whether to apply occlusion simulation for LiDAR (default: False)
            **kwargs: Additional arguments passed to BaseDataset
        """
        self.total_dataset_size = dataset_size
        self.overlap_range = tuple(overlap_range)
        self.matching_radius = matching_radius
        self.rotation_mag = rotation_mag
        self.translation_mag = translation_mag
        self.min_points = min_points
        self.max_trials = max_trials
        self.cache_filepath = cache_filepath
        
        # Validate crop method - only LiDAR is supported
        assert crop_method == 'lidar', f"crop_method must be 'lidar', got '{crop_method}'"
        self.crop_method = crop_method
        
        # Store LiDAR parameters (temporarily force FOV-only cropping)
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_horizontal_fov = float(lidar_horizontal_fov)
        self.lidar_vertical_fov = float(lidar_vertical_fov)
        # TEMPORARY: Force FOV-only cropping, disable range and occlusion filters
        self.lidar_apply_range_filter = False
        self.lidar_apply_fov_filter = True
        self.lidar_apply_occlusion_filter = False
        
        # Initialize transform-to-overlap cache
        if cache_filepath is not None:
            assert cache_filepath.endswith(".json"), f"Cache file must be JSON format, got: {cache_filepath}"
            # Ensure parent directory can be created
            cache_dir = os.path.dirname(cache_filepath)
            if cache_dir:  # Only create if there's actually a directory part
                os.makedirs(cache_dir, exist_ok=True)
            self._load_transform_cache()
        else:
            # No caching
            self.transform_cache = {}
        
        # Thread safety for parallel processing
        self._cache_lock = threading.Lock()
        
        super().__init__(data_root=data_root, **kwargs)
        
        # Calculate pairs per source file after annotations are initialized
        self._calculate_pairs_per_file()
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations - to be implemented by subclasses.
        
        Subclasses should:
        1. Set self.file_pair_annotations to list of file pair annotations
        2. Each annotation should have 'src_filepath' and 'tgt_filepath' keys
        3. For single-temporal: src_filepath == tgt_filepath  
        4. For bi-temporal: src_filepath != tgt_filepath
        """
        raise NotImplementedError("Subclasses must implement _init_annotations and set self.file_pair_annotations")
    
    def _calculate_pairs_per_file(self) -> None:
        """Calculate how many synthetic pairs to generate per file pair."""
        num_file_pairs = len(self.file_pair_annotations)
        base_pairs_per_file = self.total_dataset_size // num_file_pairs
        remainder = self.total_dataset_size % num_file_pairs
        
        # Distribute pairs evenly with remainder distributed to first files
        self.pairs_per_file = []
        for i in range(num_file_pairs):
            extra_pair = 1 if i < remainder else 0
            self.pairs_per_file.append(base_pairs_per_file + extra_pair)
        
        # Create flat annotations mapping each datapoint to (file_idx, pair_idx)
        self.annotations = []
        for file_idx, num_pairs in enumerate(self.pairs_per_file):
            for pair_idx in range(num_pairs):
                self.annotations.append({
                    'file_idx': file_idx,
                    'pair_idx': pair_idx,
                    'file_pair_annotation': self.file_pair_annotations[file_idx]
                })
    
    def _load_transform_cache(self) -> None:
        """Load cached transform-to-overlap mappings."""
        if os.path.exists(self.cache_filepath):
            with open(self.cache_filepath, 'r') as f:
                content = f.read().strip()
                if content:  # Only try to parse if file is not empty
                    loaded_cache = json.loads(content)
                    
                    # Validate cache structure - will raise AssertionError if invalid
                    self._validate_cache_structure(loaded_cache)
                    
                    # Convert string keys back to tuples for in-memory use
                    self.transform_cache = {}
                    for key_str, file_data in loaded_cache.items():
                        param_tuple = eval(key_str)  # Convert string back to tuple
                        self.transform_cache[param_tuple] = file_data
                else:
                    self.transform_cache = {}
        else:
            self.transform_cache = {}
    
    def _validate_cache_structure(self, cache_data: Any) -> None:
        """Validate that cache has the expected structure using assertions.
        
        Expected structure (JSON uses string representations of tuples):
        {
            "(rotation_mag, translation_mag, matching_radius)": {
                "file_hash_1": [
                    {
                        "overlap": float,
                        "rotation_angles": [float, float, float],
                        "translation": [float, float, float],
                        "crop_method": str,
                        "keep_ratio": float,
                        "src_num_points": int,
                        "tgt_num_points": int,
                        ...
                    },
                    ...
                ],
                ...
            },
            ...
        }
        
        Args:
            cache_data: Data loaded from cache file
        """
        # Check if cache_data is a dictionary
        assert isinstance(cache_data, dict), f"Cache data must be a dictionary, got {type(cache_data)}"
        
        # Check each parameter tuple level
        for param_key, param_data in cache_data.items():
            # Parameter key should be a string representation of a tuple
            assert isinstance(param_key, str), f"Parameter key must be string, got {type(param_key)}"
            
            # Validate parameter key format: "(rotation_mag, translation_mag, matching_radius)"
            assert param_key.startswith("(") and param_key.endswith(")"), f"Parameter key must be tuple format, got {param_key}"
            # Try to evaluate the string as a tuple
            param_tuple = eval(param_key)
            assert isinstance(param_tuple, tuple), f"Parameter key must evaluate to tuple, got {type(param_tuple)}"
            assert len(param_tuple) == 3, f"Parameter tuple must have 3 values, got {len(param_tuple)}: {param_tuple}"
            # Validate that all elements are numbers
            for i, val in enumerate(param_tuple):
                assert isinstance(val, (int, float)), f"Parameter tuple element {i} must be numeric, got {type(val)}: {val}"
            
            # Parameter data should be a dictionary
            assert isinstance(param_data, dict), f"Parameter data must be dictionary, got {type(param_data)}"
            
            # Check each file_hash level
            for file_key, transforms in param_data.items():
                # File key should be a string (hash)
                assert isinstance(file_key, str), f"File key must be string, got {type(file_key)}"
                
                # Transforms should be a list
                assert isinstance(transforms, list), f"Transforms must be list, got {type(transforms)}"
                
                # Check each transform config
                for i, transform in enumerate(transforms):
                    assert isinstance(transform, dict), f"Transform {i} must be dict, got {type(transform)}"
                    
                    # Check required fields (basic fields that all methods have)
                    basic_required_fields = ['overlap', 'rotation_angles', 'translation', 
                                           'crop_method', 'src_num_points', 'tgt_num_points']
                    
                    for field in basic_required_fields:
                        assert field in transform, f"Transform {i} missing required field '{field}'"
                    
                    # Check LiDAR-specific required fields (LiDAR is the only supported method)
                    lidar_fields = ['sensor_position', 'sensor_euler_angles', 'lidar_max_range',
                                  'lidar_horizontal_fov', 'lidar_vertical_fov', 'crop_seed']
                    for field in lidar_fields:
                        assert field in transform, f"Transform {i} missing required LiDAR field '{field}'"
                    
                    # Validate field types and values
                    assert isinstance(transform['overlap'], (int, float)), f"overlap must be number, got {type(transform['overlap'])}"
                    
                    assert isinstance(transform['rotation_angles'], list), f"rotation_angles must be list, got {type(transform['rotation_angles'])}"
                    assert len(transform['rotation_angles']) == 3, f"rotation_angles must have 3 elements, got {len(transform['rotation_angles'])}"
                    
                    assert isinstance(transform['translation'], list), f"translation must be list, got {type(transform['translation'])}"
                    assert len(transform['translation']) == 3, f"translation must have 3 elements, got {len(transform['translation'])}"
                    
                    
                    assert isinstance(transform['src_num_points'], int), f"src_num_points must be int, got {type(transform['src_num_points'])}"
                    assert isinstance(transform['tgt_num_points'], int), f"tgt_num_points must be int, got {type(transform['tgt_num_points'])}"
    
    def _save_transform_cache(self) -> None:
        """Save transform-to-overlap mappings to cache (thread-safe)."""
        if self.cache_filepath is not None:
            os.makedirs(os.path.dirname(self.cache_filepath), exist_ok=True)
            # Convert tuple keys to strings for JSON serialization
            serializable_cache = {}
            for param_tuple, file_data in self.transform_cache.items():
                key_str = str(param_tuple)  # Convert tuple to string
                serializable_cache[key_str] = file_data
            
            with open(self.cache_filepath, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
    
    def _get_file_cache_key(self, file_pair_annotation: Dict[str, Any]) -> str:
        """Generate cache key for file pair.
        
        Uses both source and target file paths to ensure unique keys
        for multi-pairing scenarios like (A,B), (A,C), (A,D).
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Unique cache key string for this file pair
        """
        src_path = file_pair_annotation['src_filepath']
        tgt_path = file_pair_annotation['tgt_filepath']
        combined_path = f"{src_path}|{tgt_path}"
        file_hash = hashlib.md5(combined_path.encode()).hexdigest()[:8]
        return file_hash
    
    def _get_cache_param_key(self) -> Tuple:
        """Generate cache parameter key for transform caching.
        
        Can be overridden by subclasses to include additional parameters.
        
        Returns:
            Cache parameter key tuple
        """
        return (self.rotation_mag, self.translation_mag, self.matching_radius)
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a synthetic datapoint using the modular pipeline.
        
        Clean flow: generate transforms if needed, then get specific transform.
        """
        file_idx, transform_idx = self._get_indices(idx)
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Get cache key for this file pair
        file_cache_key = self._get_file_cache_key(file_pair_annotation)
        
        # Helper function to get valid transforms from cache
        def get_valid_transforms():
            # Access cache with parameter tuple as outer key
            param_key = self._get_cache_param_key()
            param_cache = self.transform_cache.get(param_key, {})
            cached_transforms = param_cache.get(file_cache_key, [])
            
            valid_transforms = []
            for transform_params in cached_transforms:
                overlap = transform_params.get('overlap', 0.0)
                src_num_points = transform_params.get('src_num_points', 0)
                tgt_num_points = transform_params.get('tgt_num_points', 0)
                
                # Filter by overlap range and minimum points
                if (self.overlap_range[0] < overlap <= self.overlap_range[1] and
                    src_num_points >= self.min_points and tgt_num_points >= self.min_points):
                    valid_transforms.append(transform_params)
            return valid_transforms
        
        # Check if we have enough valid transforms
        valid_transforms = get_valid_transforms()
        if transform_idx >= len(valid_transforms):
            # Generate more transforms to fill the cache
            needed_count = transform_idx - len(valid_transforms) + 1
            self._generate_more(file_idx, needed_count)
            # Refresh valid transforms after generation
            valid_transforms = get_valid_transforms()
            
            # Check if generation was successful
            if transform_idx >= len(valid_transforms):
                raise RuntimeError(
                    f"Failed to generate enough valid transforms for datapoint index {idx}. "
                    f"Requested transform_idx={transform_idx}, but only {len(valid_transforms)} valid transforms available. "
                    f"file_idx={file_idx}, needed_count={needed_count}. "
                    f"Consider: 1) Increasing max_trials (current: {self.max_trials}), "
                    f"2) Relaxing overlap_range (current: {self.overlap_range}), "
                    f"3) Reducing min_points (current: {self.min_points}), "
                    f"4) Reducing dataset_size to match available valid transforms."
                )
        
        # Get the specific transform (called only once)
        src_pc, tgt_pc, transform_matrix, transform_params = self._get_pair(file_idx, transform_idx, valid_transforms)
        
        # Find correspondences
        from utils.point_cloud_ops.correspondences import get_correspondences
        correspondences = get_correspondences(
            src_points=src_pc['pos'],
            tgt_points=tgt_pc['pos'], 
            transform=transform_matrix,
            radius=self.matching_radius,
        )
        
        # Add default features if not present (should be done in _load_datapoint)
        if 'feat' not in src_pc:
            src_pc['feat'] = torch.ones((src_pc['pos'].shape[0], 1), dtype=torch.float32, device=src_pc['pos'].device)
        
        if 'feat' not in tgt_pc:
            tgt_pc['feat'] = torch.ones((tgt_pc['pos'].shape[0], 1), dtype=torch.float32, device=tgt_pc['pos'].device)

        inputs = {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': correspondences,
        }
        
        labels = {
            'transform': transform_matrix,
        }
        
        # Create clean transform_params without overlap (to avoid duplication with meta_info.overlap)
        clean_transform_params = {k: v for k, v in transform_params.items() if k != 'overlap'}
        
        meta_info = {
            'file_idx': file_idx,
            'transform_idx': transform_idx,
            'transform_params': clean_transform_params,
            'overlap': transform_params['overlap'],  # Keep in meta_info for easy access
        }
        
        return inputs, labels, meta_info

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load point cloud data for both source and target files.
        
        Handles both single-temporal and bi-temporal datasets:
        - Single-temporal: src_filepath == tgt_filepath, load once and copy
        - Bi-temporal: src_filepath != tgt_filepath, load both files
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_data, tgt_pc_data) point cloud dictionaries with all attributes (pos, rgb, etc.)
        """
        src_filepath = file_pair_annotation['src_filepath']
        tgt_filepath = file_pair_annotation['tgt_filepath']
        
        # Load source point cloud (load_point_cloud now always returns dict format)
        src_pc_data = load_point_cloud(src_filepath, device=self.device)
        
        # Check if single-temporal or bi-temporal
        if src_filepath == tgt_filepath:
            # Single-temporal: deep copy source as target to avoid reference issues
            tgt_pc_data = {}
            for key, value in src_pc_data.items():
                if isinstance(value, torch.Tensor):
                    tgt_pc_data[key] = value.clone()
                else:
                    tgt_pc_data[key] = value
        else:
            # Bi-temporal: load target separately
            tgt_pc_data = load_point_cloud(tgt_filepath, device=self.device)
        
        return src_pc_data, tgt_pc_data
    
    def _get_indices(self, idx: int) -> Tuple[int, int]:
        """Get file index and transform index from dataset index.
        
        Args:
            idx: Dataset index
            
        Returns:
            Tuple of (file_idx, transform_idx)
        """
        annotation = self.annotations[idx]
        file_idx = annotation['file_idx']
        transform_idx = annotation['pair_idx']
        return file_idx, transform_idx
    
    def _get_pair(self, file_idx: int, transform_idx: int, valid_transforms: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Get a transformed point cloud pair using cached transform params.
        
        Args:
            file_idx: Index of file pair
            transform_idx: Index of transform for this file pair
            valid_transforms: List of valid cached transforms for this file pair
            
        Returns:
            Tuple of (src_pc, tgt_pc, transform_matrix, transform_params)
        """
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Load point cloud data - handles both single-temporal and bi-temporal
        src_pc_data, tgt_pc_data = self._load_file_pair_data(file_pair_annotation)
        
        # Get the specific transform config from cache
        transform_params = valid_transforms[transform_idx]
        
        # Build transform components from cached params
        transform_matrix, crop_transform = self._build_transform(transform_params)
        
        # Apply transform to get point cloud pair
        src_pc, tgt_pc = self._apply_transform(src_pc_data, tgt_pc_data, transform_matrix, crop_transform, transform_params)
        
        return src_pc, tgt_pc, transform_matrix, transform_params
    
    def _generate_more(self, file_idx: int, needed_count: int) -> None:
        """Generate more valid transforms and cache them.
        
        Args:
            file_idx: Index of file pair
            needed_count: Number of additional valid transforms needed
        """
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Load point cloud data - handles both single-temporal and bi-temporal
        src_pc_data, tgt_pc_data = self._load_file_pair_data(file_pair_annotation)
        
        # Get cache key for this file pair
        file_cache_key = self._get_file_cache_key(file_pair_annotation)
        
        # Get existing cached transforms (thread-safe read)
        with self._cache_lock:
            param_key = self._get_cache_param_key()
            param_cache = self.transform_cache.get(param_key, {})
            cached_transforms = param_cache.get(file_cache_key, []).copy()
        
        generated_results = []
        trial = len(cached_transforms)  # Start from where cache left off
        target_attempts = needed_count * 3  # Generate more than needed for better hit rate
        
        while len(generated_results) < needed_count and trial < self.max_trials:
            # Process single transform (deterministic seed using file_idx, trial)
            result = self._process_single_transform((src_pc_data, tgt_pc_data, file_idx, trial))
            
            # Update cache with new transform
            cached_transforms.append(result['transform_params'])
            
            # Check if overlap is in range and meets minimum points requirement
            src_num_points = result['transform_params']['src_num_points']
            tgt_num_points = result['transform_params']['tgt_num_points']
            if (self.overlap_range[0] < result['overlap'] <= self.overlap_range[1] and
                src_num_points >= self.min_points and tgt_num_points >= self.min_points):
                generated_results.append((
                    result['src_pc'], 
                    result['tgt_pc'], 
                    result['transform_matrix'], 
                    result['transform_params']
                ))
            
            trial += 1
            
            # Periodically save cache to avoid losing progress
            if trial % 10 == 0:
                with self._cache_lock:
                    param_key = self._get_cache_param_key()
                    if param_key not in self.transform_cache:
                        self.transform_cache[param_key] = {}
                    self.transform_cache[param_key][file_cache_key] = cached_transforms.copy()
                    if self.cache_filepath is not None:
                        self._save_transform_cache()
            
            # Early exit if we have enough valid results and attempted enough
            if len(generated_results) >= needed_count or (trial - len(cached_transforms)) >= target_attempts:
                break
        
        # Final cache update
        with self._cache_lock:
            param_key = self._get_cache_param_key()
            if param_key not in self.transform_cache:
                self.transform_cache[param_key] = {}
            self.transform_cache[param_key][file_cache_key] = cached_transforms
            if self.cache_filepath is not None:
                self._save_transform_cache()
        
        # Fail fast if no valid results found - don't hide the problem
        assert len(generated_results) > 0, (
            f"Failed to generate any valid transforms after {trial} trials. "
            f"Parameters: overlap_range={self.overlap_range}, file_idx={file_idx}, "
            f"needed_count={needed_count}. "
            f"Consider: 1) Relaxing overlap_range, 2) Reducing cropping aggressiveness, "
            f"3) Adjusting rotation/translation ranges for ModelNet40 object scale."
        )
    
    def _process_single_transform(self, args: Tuple) -> Dict[str, Any]:
        """Process a single transform.
        
        Args:
            args: Tuple of (src_pc_data, tgt_pc_data, file_idx, trial_idx)
            
        Returns:
            Result dictionary with transform data
        """
        src_pc_data, tgt_pc_data, file_idx, trial_idx = args
        
        # Create deterministic seed from (file_idx, trial_idx)
        # file_idx already provides uniqueness for multi-pairing scenarios
        seed = hash((file_idx, trial_idx)) % (2**32)
        
        # Sample transform parameters (deterministic from seed)
        transform_params = self._sample_transform(seed)
        
        # Build transform components
        transform_matrix, crop_transform = self._build_transform(transform_params)
        
        # Apply transform
        src_pc, tgt_pc = self._apply_transform(src_pc_data, tgt_pc_data, transform_matrix, crop_transform, transform_params)
        
        # Compute overlap (this is the expensive operation we're parallelizing)
        # Following standard PCR convention: compute overlap between source + transform vs target
        # This measures how well the source aligns to target with the ground truth transform
        overlap = compute_registration_overlap(
            ref_points=tgt_pc['pos'],
            src_points=src_pc['pos'], 
            transform=transform_matrix,
            positive_radius=self.matching_radius * 2
        )
        
        # Add metadata including point counts for min_points filtering
        transform_params['overlap'] = float(overlap)
        transform_params['src_num_points'] = src_pc['pos'].shape[0]
        transform_params['tgt_num_points'] = tgt_pc['pos'].shape[0]
        transform_params['trial'] = trial_idx
        
        return {
            'transform_params': transform_params,
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'transform_matrix': transform_matrix,
            'overlap': overlap,
            'seed': seed
        }
    
    def _sample_transform(self, seed: int) -> Dict[str, Any]:
        """Sample transform parameters with configurable cropping method.
        
        Args:
            seed: Random seed for deterministic sampling
            
        Returns:
            Dictionary containing all transform parameters including crop parameters
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Sample SE(3) transformation parameters for creating synthetic misaligned pairs
        rotation_angles = torch.rand(3, generator=generator) * (2 * self.rotation_mag) - self.rotation_mag  # [-rotation_mag, rotation_mag] degrees
        translation = torch.rand(3, generator=generator) * (2 * self.translation_mag) - self.translation_mag  # [-translation_mag, translation_mag]
        
        # Only LiDAR cropping is supported
        # Sample sensor pose for LiDAR simulation
        # Position: sample within a reasonable range around the point cloud
        sensor_position = torch.rand(3, generator=generator) * 10.0 - 5.0  # [-5, 5] range
        
        # Rotation: sample random orientation using Euler angles
        euler_angles = torch.rand(3, generator=generator) * 2 * np.pi  # [0, 2π] range
        
        # Generate crop seed for deterministic cropping (derived from main seed)
        crop_seed = (seed * 31 + 42) % (2**32)  # Deterministic derivation from main seed
        
        # Convert to 4x4 extrinsics matrix parameters for deterministic reconstruction
        config = {
            'rotation_angles': rotation_angles.tolist(),
            'translation': translation.tolist(),
            'crop_method': 'lidar',
            'sensor_position': sensor_position.tolist(),
            'sensor_euler_angles': euler_angles.tolist(),
            'lidar_max_range': self.lidar_max_range,
            'lidar_horizontal_fov': self.lidar_horizontal_fov,
            'lidar_vertical_fov': self.lidar_vertical_fov,
            'seed': seed,
            'crop_seed': crop_seed,
        }
        
        return config
    
    def _build_transform(self, transform_params: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """Build transform matrix and crop object from parameters.
        
        Args:
            transform_params: Transform configuration dictionary
            
        Returns:
            Tuple of (transform_matrix, crop_transform)
        """
        # Build SE(3) transformation matrix
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
        
        # Build LiDAR crop transform using pre-sampled parameters for deterministic caching
        assert transform_params['crop_method'] == 'lidar', f"Only LiDAR crop method is supported, got '{transform_params['crop_method']}'"
        
        # Build 4x4 extrinsics matrix from sampled pose parameters
        sensor_position = torch.tensor(transform_params['sensor_position'], dtype=torch.float32, device=self.device)
        euler_angles = torch.tensor(transform_params['sensor_euler_angles'], dtype=torch.float32, device=self.device)
        
        # Convert Euler angles to rotation matrix
        cos_vals = torch.cos(euler_angles)
        sin_vals = torch.sin(euler_angles)
        
        # ZYX Euler angle convention (yaw, pitch, roll)
        R_z = torch.tensor([
            [cos_vals[2], -sin_vals[2], 0],
            [sin_vals[2], cos_vals[2], 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        R_y = torch.tensor([
            [cos_vals[1], 0, sin_vals[1]],
            [0, 1, 0],
            [-sin_vals[1], 0, cos_vals[1]]
        ], dtype=torch.float32, device=self.device)
        
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_vals[0], -sin_vals[0]],
            [0, sin_vals[0], cos_vals[0]]
        ], dtype=torch.float32, device=self.device)
        
        rotation_matrix = R_z @ R_y @ R_x
        
        # Build 4x4 extrinsics matrix
        sensor_extrinsics = torch.eye(4, dtype=torch.float32, device=self.device)
        sensor_extrinsics[:3, :3] = rotation_matrix
        sensor_extrinsics[:3, 3] = sensor_position
        
        # Create LiDAR crop transform with FOV-only settings (temporarily forced)
        crop_transform = LiDARSimulationCrop(
            max_range=transform_params['lidar_max_range'],
            horizontal_fov=transform_params['lidar_horizontal_fov'],
            vertical_fov=transform_params['lidar_vertical_fov'],
            # TEMPORARY: Force FOV-only cropping, disable range and occlusion filters
            apply_range_filter=False,
            apply_fov_filter=True,
            apply_occlusion_filter=False
        )
        
        # Store sensor extrinsics for LiDAR cropping
        crop_transform._sensor_extrinsics = sensor_extrinsics
        
        return transform_matrix, crop_transform
    
    def _apply_transform(self, src_pc_data: Dict[str, torch.Tensor], tgt_pc_data: Dict[str, torch.Tensor], 
                        transform_matrix: torch.Tensor, crop_transform: Any, 
                        transform_params: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply transform to create source and target point clouds.
        
        Standard PCR convention: source + gt_transform = target
        Following GeoTransformer's approach:
        1. ref_points (target) = original point cloud 
        2. src_points (source) = apply inverse transform + crop to create misaligned source
        3. transform = forward transform that aligns src back to ref
        
        Args:
            src_pc_data: Raw source point cloud dictionary (pos, rgb, etc.)
            tgt_pc_data: Raw target point cloud dictionary (same as src_pc_data for single-temporal)
            transform_matrix: 4x4 transformation matrix to align source to target
            crop_transform: Crop transform object
            transform_params: Transform configuration
            
        Returns:
            Tuple of (src_pc, tgt_pc) dictionaries where apply_transform(src_pc, transform_matrix) ≈ tgt_pc
        """
        # Following GeoTransformer's approach:
        # ref_points = original (target)
        ref_points = src_pc_data['pos'].clone()
        
        # src_points = apply inverse transform to create misaligned source
        from utils.point_cloud_ops import apply_transform
        # Create a fresh tensor to avoid threading issues
        transform_inv = torch.linalg.inv(transform_matrix.detach().cpu()).to(transform_matrix.device)
        src_points = apply_transform(ref_points, transform_inv)
        
        # Apply LiDAR cropping to both source and target 
        assert transform_params['crop_method'] == 'lidar', f"Only LiDAR crop method is supported, got '{transform_params['crop_method']}'"
        
        # LiDAR cropping requires sensor extrinsics matrix
        sensor_extrinsics = crop_transform._sensor_extrinsics
        
        # Update point cloud dictionaries with transformed/original positions
        src_pc_data_transformed = src_pc_data.copy()
        src_pc_data_transformed['pos'] = src_points
        
        tgt_pc_data_original = tgt_pc_data.copy()
        tgt_pc_data_original['pos'] = ref_points
        
        # Apply LiDAR cropping (preserves all keys including RGB)
        # NOTE: LiDARSimulationCrop has custom signature requiring sensor_extrinsics parameter
        # that doesn't fit standard BaseTransform API, so we must use _call_single
        # Use crop_seed from transform_params for deterministic cropping
        crop_seed = transform_params['crop_seed']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(crop_seed)
        src_pc_dict = crop_transform._call_single(src_pc_data_transformed, sensor_extrinsics, generator=generator)
        tgt_pc_dict = crop_transform._call_single(tgt_pc_data_original, sensor_extrinsics, generator=generator)
        
        # Return the cropped point cloud dictionaries (already contains pos, rgb if available, etc.)
        return src_pc_dict, tgt_pc_dict
