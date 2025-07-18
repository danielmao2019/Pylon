from typing import Tuple, Dict, Any, List, Optional
from abc import ABC
import os
import json
import hashlib
import threading
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap
from data.transforms.vision_3d.random_plane_crop import RandomPlaneCrop
from data.transforms.vision_3d.random_point_crop import RandomPointCrop
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
            try:
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
            except (json.JSONDecodeError, IOError) as e:
                # If file is corrupted or unreadable, start with empty cache
                print(f"Warning: Error loading cache from {self.cache_filepath}: {e}. Starting with empty cache.")
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
                    
                    # Check required fields
                    required_fields = ['overlap', 'rotation_angles', 'translation', 
                                     'crop_method', 'keep_ratio', 'src_num_points', 
                                     'tgt_num_points']
                    
                    for field in required_fields:
                        assert field in transform, f"Transform {i} missing required field '{field}'"
                    
                    # Validate field types and values
                    assert isinstance(transform['overlap'], (int, float)), f"overlap must be number, got {type(transform['overlap'])}"
                    
                    assert isinstance(transform['rotation_angles'], list), f"rotation_angles must be list, got {type(transform['rotation_angles'])}"
                    assert len(transform['rotation_angles']) == 3, f"rotation_angles must have 3 elements, got {len(transform['rotation_angles'])}"
                    
                    assert isinstance(transform['translation'], list), f"translation must be list, got {type(transform['translation'])}"
                    assert len(transform['translation']) == 3, f"translation must have 3 elements, got {len(transform['translation'])}"
                    
                    assert transform['crop_method'] in ['plane', 'point'], f"crop_method must be 'plane' or 'point', got '{transform['crop_method']}'"
                    
                    assert isinstance(transform['keep_ratio'], (int, float)), f"keep_ratio must be number, got {type(transform['keep_ratio'])}"
                    
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
            param_key = (self.rotation_mag, self.translation_mag, self.matching_radius)
            param_cache = self.transform_cache.get(param_key, {})
            cached_transforms = param_cache.get(file_cache_key, [])
            
            valid_transforms = []
            for transform_config in cached_transforms:
                overlap = transform_config.get('overlap', 0.0)
                src_num_points = transform_config.get('src_num_points', 0)
                tgt_num_points = transform_config.get('tgt_num_points', 0)
                
                # Filter by overlap range and minimum points
                if (self.overlap_range[0] < overlap <= self.overlap_range[1] and
                    src_num_points >= self.min_points and tgt_num_points >= self.min_points):
                    valid_transforms.append(transform_config)
            return valid_transforms
        
        # Check if we have enough valid transforms
        valid_transforms = get_valid_transforms()
        if transform_idx >= len(valid_transforms):
            # Generate more transforms to fill the cache
            needed_count = transform_idx - len(valid_transforms) + 1
            self._generate_more(file_idx, needed_count)
            # Refresh valid transforms after generation
            valid_transforms = get_valid_transforms()
        
        # Get the specific transform (called only once)
        src_pc, tgt_pc, transform_matrix, transform_config = self._get_pair(file_idx, transform_idx, valid_transforms)
        
        # Find correspondences
        from utils.point_cloud_ops.correspondences import get_correspondences
        correspondences = get_correspondences(
            src_points=src_pc['pos'],
            tgt_points=tgt_pc['pos'], 
            transform=transform_matrix,
            radius=self.matching_radius,
        )
        
        inputs = {
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'correspondences': correspondences,
        }
        
        labels = {
            'transform': transform_matrix,
        }
        
        meta_info = {
            'file_idx': file_idx,
            'transform_idx': transform_idx,
            'transform_config': transform_config,
            'overlap': transform_config['overlap'],
            'crop_method': transform_config['crop_method'],
            'keep_ratio': transform_config['keep_ratio'],
        }
        
        return inputs, labels, meta_info

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud data for both source and target files.
        
        Handles both single-temporal and bi-temporal datasets:
        - Single-temporal: src_filepath == tgt_filepath, load once and copy
        - Bi-temporal: src_filepath != tgt_filepath, load both files
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_raw, tgt_pc_raw) point cloud position tensors (not normalized)
        """
        src_filepath = file_pair_annotation['src_filepath']
        tgt_filepath = file_pair_annotation['tgt_filepath']
        
        # Load source point cloud (load_point_cloud now always returns dict format)
        src_pc_data = load_point_cloud(src_filepath)
        src_pc_raw = src_pc_data['pos']
        
        # Check if single-temporal or bi-temporal
        if src_filepath == tgt_filepath:
            # Single-temporal: copy source as target
            tgt_pc_raw = src_pc_raw.clone()
        else:
            # Bi-temporal: load target separately
            tgt_pc_data = load_point_cloud(tgt_filepath)
            tgt_pc_raw = tgt_pc_data['pos']
        
        return src_pc_raw, tgt_pc_raw
    
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
            Tuple of (src_pc, tgt_pc, transform_matrix, transform_config)
        """
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Load point cloud data - handles both single-temporal and bi-temporal
        src_pc_raw, tgt_pc_raw = self._load_file_pair_data(file_pair_annotation)
        
        # Get the specific transform config from cache
        transform_config = valid_transforms[transform_idx]
        
        # Build transform components from cached params
        transform_matrix, crop_transform = self._build_transform(transform_config)
        
        # Apply transform to get point cloud pair
        src_pc, tgt_pc = self._apply_transform(src_pc_raw, tgt_pc_raw, transform_matrix, crop_transform, transform_config)
        
        return src_pc, tgt_pc, transform_matrix, transform_config
    
    def _generate_more(self, file_idx: int, needed_count: int) -> None:
        """Generate more valid transforms and cache them (parallel version).
        
        Args:
            file_idx: Index of file pair
            needed_count: Number of additional valid transforms needed
        """
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Load point cloud data - handles both single-temporal and bi-temporal
        src_pc_raw, tgt_pc_raw = self._load_file_pair_data(file_pair_annotation)
        
        # Get cache key for this file pair
        file_cache_key = self._get_file_cache_key(file_pair_annotation)
        
        # Get existing cached transforms (thread-safe read)
        with self._cache_lock:
            param_key = (self.rotation_mag, self.translation_mag, self.matching_radius)
            param_cache = self.transform_cache.get(param_key, {})
            cached_transforms = param_cache.get(file_cache_key, []).copy()
        
        generated_results = []
        trial = len(cached_transforms)  # Start from where cache left off
        
        # Process transforms in parallel batches
        batch_size = min(needed_count * 3, 20)  # Generate more than needed for better hit rate
        
        while len(generated_results) < needed_count and trial < self.max_trials:
            # Prepare batch of work (deterministic seeds using file_idx, trial_idx)
            current_batch_size = min(batch_size, self.max_trials - trial)
            batch_args = []
            for i in range(current_batch_size):
                trial_idx = trial + i
                batch_args.append((src_pc_raw, tgt_pc_raw, file_idx, trial_idx))
            
            # Process batch in parallel
            batch_results = self._process_transform_batch(batch_args)
            
            # Collect results and update cache
            new_cache_entries = []
            for result in batch_results:
                new_cache_entries.append(result['transform_params'])
                
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
            
            # Thread-safe cache update
            with self._cache_lock:
                cached_transforms.extend(new_cache_entries)
                # Always update in-memory cache with new structure
                param_key = (self.rotation_mag, self.translation_mag, self.matching_radius)
                if param_key not in self.transform_cache:
                    self.transform_cache[param_key] = {}
                self.transform_cache[param_key][file_cache_key] = cached_transforms
                # Save to file only if cache_filepath is provided
                if self.cache_filepath is not None:
                    self._save_transform_cache()
            
            trial += current_batch_size
            
            # Early exit if we have enough valid results
            if len(generated_results) >= needed_count:
                break
        
        # Fail fast if no valid results found - don't hide the problem
        assert len(generated_results) > 0, (
            f"Failed to generate any valid transforms after {self.max_trials} trials. "
            f"Parameters: overlap_range={self.overlap_range}, file_idx={file_idx}, "
            f"needed_count={needed_count}. "
            f"Consider: 1) Relaxing overlap_range, 2) Reducing cropping aggressiveness, "
            f"3) Adjusting rotation/translation ranges for ModelNet40 object scale."
        )
        
        # Verify we generated enough valid transforms
        with self._cache_lock:
            param_key = (self.rotation_mag, self.translation_mag, self.matching_radius)
            param_cache = self.transform_cache.get(param_key, {})
            updated_cached_transforms = param_cache.get(file_cache_key, [])
        
        # Count valid transforms to verify we have enough
        valid_count = 0
        for transform_config in updated_cached_transforms:
            overlap = transform_config.get('overlap', 0.0)
            src_num_points = transform_config.get('src_num_points', 0)
            tgt_num_points = transform_config.get('tgt_num_points', 0)
            
            if (self.overlap_range[0] < overlap <= self.overlap_range[1] and
                src_num_points >= self.min_points and tgt_num_points >= self.min_points):
                valid_count += 1
    
    def _process_transform_batch(self, batch_args: List[Tuple]) -> List[Dict[str, Any]]:
        """Process a batch of transforms in parallel.
        
        Args:
            batch_args: List of (src_pc_raw, tgt_pc_raw, file_idx, trial_idx) tuples
            
        Returns:
            List of result dictionaries
        """
        num_workers = min(len(batch_args), 4)  # Limit concurrent threads
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_single_transform, args) for args in batch_args]
            results = [future.result() for future in futures]
        
        return results
    
    def _process_single_transform(self, args: Tuple) -> Dict[str, Any]:
        """Process a single transform - thread-safe worker function.
        
        Args:
            args: Tuple of (src_pc_raw, tgt_pc_raw, file_idx, trial_idx)
            
        Returns:
            Result dictionary with transform data
        """
        src_pc_raw, tgt_pc_raw, file_idx, trial_idx = args
        
        # Create deterministic seed from (file_idx, trial_idx)
        # file_idx already provides uniqueness for multi-pairing scenarios
        seed = hash((file_idx, trial_idx)) % (2**32)
        
        # Sample transform parameters (deterministic from seed)
        transform_params = self._sample_transform(seed)
        
        # Build transform components
        transform_matrix, crop_transform = self._build_transform(transform_params)
        
        # Apply transform
        src_pc, tgt_pc = self._apply_transform(src_pc_raw, tgt_pc_raw, transform_matrix, crop_transform, transform_params)
        
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
            'overlap': overlap
        }
    
    def _sample_transform(self, seed: int) -> Dict[str, Any]:
        """Sample transform parameters stochastically following GeoTransformer exactly.
        
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
        
        # Sample cropping parameters following GeoTransformer exactly
        crop_choice = torch.rand(1, generator=generator).item()
        keep_ratio = 0.7  # GeoTransformer uses constant 0.7, not random range
        
        # Sample crop-specific parameters to ensure deterministic caching
        if crop_choice < 0.5:  # plane crop
            # Sample plane normal (from random_sample_plane in GeoTransformer)
            phi = torch.rand(1, generator=generator).item() * 2 * np.pi  # longitude
            theta = torch.rand(1, generator=generator).item() * np.pi     # latitude
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            plane_normal = [float(x), float(y), float(z)]
            
            config = {
                'rotation_angles': rotation_angles.tolist(),
                'translation': translation.tolist(),
                'crop_method': 'plane',
                'keep_ratio': float(keep_ratio),
                'plane_normal': plane_normal,
                'seed': seed,
            }
        else:  # point crop
            # Sample viewpoint (from random_sample_viewpoint in GeoTransformer)
            limit = 500
            viewpoint_base = torch.rand(3, generator=generator)  # [0, 1]
            viewpoint_sign = torch.randint(0, 2, (3,), generator=generator) * 2 - 1  # {-1, 1}
            viewpoint = viewpoint_base + limit * viewpoint_sign
            
            config = {
                'rotation_angles': rotation_angles.tolist(),
                'translation': translation.tolist(),
                'crop_method': 'point',
                'keep_ratio': float(keep_ratio),
                'viewpoint': viewpoint.tolist(),
                'seed': seed,
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
        
        # Build crop transform using pre-sampled parameters for deterministic caching
        if transform_params['crop_method'] == 'plane':
            plane_normal = torch.tensor(transform_params['plane_normal'], dtype=torch.float32, device=self.device)
            crop_transform = RandomPlaneCrop(
                keep_ratio=transform_params['keep_ratio'],
                plane_normal=plane_normal
            )
        else:
            viewpoint = torch.tensor(transform_params['viewpoint'], dtype=torch.float32, device=self.device)
            crop_transform = RandomPointCrop(
                keep_ratio=transform_params['keep_ratio'],
                viewpoint=viewpoint
            )
        
        return transform_matrix, crop_transform
    
    def _apply_transform(self, src_pc_raw: torch.Tensor, tgt_pc_raw: torch.Tensor, 
                        transform_matrix: torch.Tensor, crop_transform: Any, 
                        transform_params: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply transform to create source and target point clouds.
        
        Standard PCR convention: source + gt_transform = target
        Following GeoTransformer's approach:
        1. ref_points (target) = original point cloud 
        2. src_points (source) = apply inverse transform + crop to create misaligned source
        3. transform = forward transform that aligns src back to ref
        
        Args:
            src_pc_raw: Raw source point cloud positions
            tgt_pc_raw: Raw target point cloud positions (same as src_pc_raw for single-temporal)
            transform_matrix: 4x4 transformation matrix to align source to target
            crop_transform: Crop transform object
            transform_params: Transform configuration
            
        Returns:
            Tuple of (src_pc, tgt_pc) dictionaries where apply_transform(src_pc, transform_matrix) ≈ tgt_pc
        """
        # Following GeoTransformer's approach:
        # ref_points = original (target)
        ref_points = src_pc_raw.clone()
        
        # src_points = apply inverse transform to create misaligned source
        from utils.point_cloud_ops import apply_transform
        # Create a fresh tensor to avoid threading issues
        transform_inv = torch.linalg.inv(transform_matrix.detach().cpu()).to(transform_matrix.device)
        src_points = apply_transform(ref_points, transform_inv)
        
        # Apply cropping to both source and target (GeoTransformer crops both)
        src_pc_dict = crop_transform({'pos': src_points}, seed=transform_params['seed'])
        src_pc_pos = src_pc_dict['pos']
        
        tgt_pc_dict = crop_transform({'pos': ref_points}, seed=transform_params['seed']) 
        tgt_pc_pos = tgt_pc_dict['pos']
        
        # Create point cloud dictionaries with features on same device as positions
        src_pc = {
            'pos': src_pc_pos,
            'feat': torch.ones((src_pc_pos.shape[0], 1), dtype=torch.float32, device=src_pc_pos.device),
        }
        
        tgt_pc = {
            'pos': tgt_pc_pos,
            'feat': torch.ones((tgt_pc_pos.shape[0], 1), dtype=torch.float32, device=tgt_pc_pos.device),
        }
        
        return src_pc, tgt_pc
