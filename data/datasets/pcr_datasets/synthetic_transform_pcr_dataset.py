from typing import Tuple, Dict, Any, List
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


class SyntheticTransformPCRDataset(BaseDataset):
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
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        matching_radius: float = 0.05,
        cache_transforms: bool = True,
        **kwargs,
    ) -> None:
        """Initialize synthetic transform PCR dataset.
        
        Args:
            data_root: Path to dataset root directory
            dataset_size: Total number of synthetic pairs to generate
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            matching_radius: Radius for correspondence finding
            cache_transforms: Whether to cache transform-to-overlap mappings
            **kwargs: Additional arguments passed to BaseDataset
        """
        self.total_dataset_size = dataset_size
        self.overlap_range = tuple(overlap_range)
        self.matching_radius = matching_radius
        self.cache_transforms = cache_transforms
        
        # Initialize transform-to-overlap cache
        if self.cache_transforms:
            cache_name = f"transform_overlap_cache_{self.overlap_range[0]}_{self.overlap_range[1]}.json"
            self.cache_file = os.path.join(data_root, '..', cache_name)
            self._load_transform_cache()
        
        # Thread safety for parallel processing
        self._cache_lock = threading.Lock()
        
        super().__init__(data_root=data_root, **kwargs)
        
        # Calculate pairs per source file after annotations are initialized
        self._calculate_pairs_per_file()
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations - to be implemented by subclasses.
        
        Subclasses should:
        1. Set self.file_pair_annotations to list of file pair annotations
        2. Each annotation should have 'src_file_path' and 'tgt_file_path' keys
        3. For single-temporal: src_file_path == tgt_file_path  
        4. For bi-temporal: src_file_path != tgt_file_path
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
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.transform_cache = json.load(f)
        else:
            self.transform_cache = {}
    
    def _save_transform_cache(self) -> None:
        """Save transform-to-overlap mappings to cache (thread-safe)."""
        if self.cache_transforms:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.transform_cache, f, indent=2)
    
    def _get_file_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return file_hash
    
    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud data for both source and target files.
        
        Handles both single-temporal and bi-temporal datasets:
        - Single-temporal: src_file_path == tgt_file_path, load once and copy
        - Bi-temporal: src_file_path != tgt_file_path, load both files
        
        Args:
            file_pair_annotation: Annotation with 'src_file_path' and 'tgt_file_path' keys
            
        Returns:
            Tuple of (src_pc_raw, tgt_pc_raw) point cloud position tensors
        """
        src_file_path = file_pair_annotation['src_file_path']
        tgt_file_path = file_pair_annotation['tgt_file_path']
        
        # Load source point cloud (load_point_cloud now always returns dict format)
        src_pc_data = load_point_cloud(src_file_path)
        src_pc_raw = src_pc_data['pos']
        
        # Check if single-temporal or bi-temporal
        if src_file_path == tgt_file_path:
            # Single-temporal: copy source as target
            tgt_pc_raw = src_pc_raw.clone()
        else:
            # Bi-temporal: load target separately
            tgt_pc_data = load_point_cloud(tgt_file_path)
            tgt_pc_raw = tgt_pc_data['pos']
        
        return src_pc_raw, tgt_pc_raw
    
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a synthetic datapoint using the modular pipeline.
        
        Checks transform-to-overlap cache first, then decides to call either _get_pair or _generate_more.
        """
        file_idx, transform_idx = self._get_indices(idx)
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Get cache key for this file pair
        file_cache_key = self._get_file_cache_key(
            file_pair_annotation.get('src_file_path', str(file_idx))
        )
        
        # Check transform-to-overlap cache
        cached_transforms = self.transform_cache.get(file_cache_key, [])
        
        # Filter valid transforms by overlap range
        valid_transforms = []
        for transform_config in cached_transforms:
            overlap = transform_config.get('overlap', 0.0)
            if self.overlap_range[0] < overlap <= self.overlap_range[1]:
                valid_transforms.append(transform_config)
        
        # Decide whether to call _get_pair or _generate_more
        if transform_idx < len(valid_transforms):
            # Found in cache - use _get_pair
            src_pc, tgt_pc, transform_matrix, transform_config = self._get_pair(file_idx, transform_idx, valid_transforms)
        else:
            # Not found in cache - use _generate_more
            needed_count = transform_idx - len(valid_transforms) + 1
            src_pc, tgt_pc, transform_matrix, transform_config = self._generate_more(file_idx, transform_idx, needed_count)
        
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
    
    def _generate_more(self, file_idx: int, transform_idx: int, needed_count: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Generate more valid transforms and return the specific one requested (parallel version).
        
        Args:
            file_idx: Index of file pair
            transform_idx: Index of transform for this file pair (unused - diagnostic warning expected)
            needed_count: Number of additional valid transforms needed
            
        Returns:
            Tuple of (src_pc, tgt_pc, transform_matrix, transform_config) for the requested transform_idx
        """
        file_pair_annotation = self.file_pair_annotations[file_idx]
        
        # Load point cloud data - handles both single-temporal and bi-temporal
        src_pc_raw, tgt_pc_raw = self._load_file_pair_data(file_pair_annotation)
        
        # Get cache key for this file pair
        file_cache_key = self._get_file_cache_key(
            file_pair_annotation.get('src_file_path', str(file_idx))
        )
        
        # Get existing cached transforms (thread-safe read)
        with self._cache_lock:
            cached_transforms = self.transform_cache.get(file_cache_key, []).copy()
        
        generated_results = []
        trial = len(cached_transforms)  # Start from where cache left off
        
        # Process transforms in parallel batches
        batch_size = min(needed_count * 3, 20)  # Generate more than needed for better hit rate
        max_trials = 1000
        
        while len(generated_results) < needed_count and trial < max_trials:
            # Prepare batch of work (deterministic seeds using file_idx, transform_idx, trial_idx)
            current_batch_size = min(batch_size, max_trials - trial)
            batch_args = []
            for i in range(current_batch_size):
                trial_idx = trial + i
                batch_args.append((src_pc_raw, tgt_pc_raw, file_idx, transform_idx, trial_idx))
            
            # Process batch in parallel
            batch_results = self._process_transform_batch(batch_args)
            
            # Collect results and update cache
            new_cache_entries = []
            for result in batch_results:
                new_cache_entries.append(result['transform_params'])
                
                # Check if overlap is in range
                if self.overlap_range[0] < result['overlap'] <= self.overlap_range[1]:
                    generated_results.append((
                        result['src_pc'], 
                        result['tgt_pc'], 
                        result['transform_matrix'], 
                        result['transform_params']
                    ))
            
            # Thread-safe cache update
            with self._cache_lock:
                cached_transforms.extend(new_cache_entries)
                if self.cache_transforms:
                    self.transform_cache[file_cache_key] = cached_transforms
                    self._save_transform_cache()
            
            trial += current_batch_size
            
            # Early exit if we have enough valid results
            if len(generated_results) >= needed_count:
                break
        
        # Return the last generated valid result
        return generated_results[-1]
    
    def _process_transform_batch(self, batch_args: List[Tuple]) -> List[Dict[str, Any]]:
        """Process a batch of transforms in parallel.
        
        Args:
            batch_args: List of (original_pc, file_idx, transform_idx, trial_idx) tuples
            
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
            args: Tuple of (src_pc_raw, tgt_pc_raw, file_idx, transform_idx, trial_idx)
            
        Returns:
            Result dictionary with transform data
        """
        src_pc_raw, tgt_pc_raw, file_idx, transform_idx, trial_idx = args
        
        # Create deterministic seed from (file_idx, transform_idx, trial_idx)
        seed = hash((file_idx, transform_idx, trial_idx)) % (2**32)
        
        # Sample transform parameters (deterministic from seed)
        transform_params = self._sample_transform(seed)
        
        # Build transform components
        transform_matrix, crop_transform = self._build_transform(transform_params)
        
        # Apply transform
        src_pc, tgt_pc = self._apply_transform(src_pc_raw, tgt_pc_raw, transform_matrix, crop_transform, transform_params)
        
        # Compute overlap (this is the expensive operation we're parallelizing)
        overlap = compute_registration_overlap(
            ref_points=tgt_pc_raw,
            src_points=src_pc['pos'],
            transform=None,
            positive_radius=self.matching_radius * 2
        )
        
        # Add metadata
        transform_params['overlap'] = float(overlap)
        transform_params['trial'] = trial_idx
        
        return {
            'transform_params': transform_params,
            'src_pc': src_pc,
            'tgt_pc': tgt_pc,
            'transform_matrix': transform_matrix,
            'overlap': overlap
        }
    
    def _sample_transform(self, seed: int) -> Dict[str, Any]:
        """Sample transform parameters stochastically.
        
        Args:
            seed: Random seed for deterministic sampling
            
        Returns:
            Dictionary containing all transform parameters
        """
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Sample SE(3) transformation parameters
        rotation_angles = torch.rand(3, generator=generator) * 60 - 30  # [-30, 30] degrees
        translation = torch.rand(3, generator=generator) * 0.6 - 0.3  # [-0.3, 0.3]
        
        # Sample cropping parameters
        crop_choice = torch.rand(1, generator=generator).item()
        keep_ratio = 0.7 + torch.rand(1, generator=generator).item() * 0.2  # [0.7, 0.9]
        
        config = {
            'rotation_angles': rotation_angles.tolist(),
            'translation': translation.tolist(),
            'crop_method': 'plane' if crop_choice < 0.5 else 'point',
            'keep_ratio': float(keep_ratio),
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
        rotation_angles = torch.tensor(transform_params['rotation_angles'], dtype=torch.float32)
        translation = torch.tensor(transform_params['translation'], dtype=torch.float32)
        
        # Convert to radians and create rotation matrices
        rotation_rad = rotation_angles * np.pi / 180
        cos_vals = torch.cos(rotation_rad)
        sin_vals = torch.sin(rotation_rad)
        
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_vals[0], -sin_vals[0]],
            [0, sin_vals[0], cos_vals[0]]
        ], dtype=torch.float32)
        
        R_y = torch.tensor([
            [cos_vals[1], 0, sin_vals[1]],
            [0, 1, 0],
            [-sin_vals[1], 0, cos_vals[1]]
        ], dtype=torch.float32)
        
        R_z = torch.tensor([
            [cos_vals[2], -sin_vals[2], 0],
            [sin_vals[2], cos_vals[2], 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        rotation_matrix = R_z @ R_y @ R_x
        
        # Create 4x4 transformation matrix
        transform_matrix = torch.eye(4, dtype=torch.float32)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation
        
        # Build crop transform
        if transform_params['crop_method'] == 'plane':
            crop_transform = RandomPlaneCrop(keep_ratio=transform_params['keep_ratio'])
        else:
            crop_transform = RandomPointCrop(keep_ratio=transform_params['keep_ratio'])
        
        return transform_matrix, crop_transform
    
    def _apply_transform(self, src_pc_raw: torch.Tensor, tgt_pc_raw: torch.Tensor, 
                        transform_matrix: torch.Tensor, crop_transform: Any, 
                        transform_params: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply transform to create source and target point clouds.
        
        Args:
            src_pc_raw: Raw source point cloud positions
            tgt_pc_raw: Raw target point cloud positions
            transform_matrix: 4x4 transformation matrix
            crop_transform: Crop transform object
            transform_params: Transform configuration
            
        Returns:
            Tuple of (src_pc, tgt_pc) dictionaries
        """
        # Apply SE(3) transformation to source
        transformed_pc = (transform_matrix[:3, :3] @ src_pc_raw.T).T + transform_matrix[:3, 3]
        
        # Apply crop with deterministic seed
        transformed_pc_dict = {'pos': transformed_pc}
        src_pc_dict = crop_transform(transformed_pc_dict, seed=transform_params['seed'])
        src_pc_pos = src_pc_dict['pos']
        
        # Create point cloud dictionaries
        src_pc = {
            'pos': src_pc_pos,
            'feat': torch.ones((src_pc_pos.shape[0], 1), dtype=torch.float32),
        }
        
        tgt_pc = {
            'pos': tgt_pc_raw,
            'feat': torch.ones((tgt_pc_raw.shape[0], 1), dtype=torch.float32),
        }
        
        return src_pc, tgt_pc
