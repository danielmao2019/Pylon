from typing import Tuple, Dict, Any, List
import os
import json
import hashlib
import numpy as np
import torch
from data.datasets.base_dataset import BaseDataset
from utils.point_cloud_ops.set_ops.intersection import compute_registration_overlap
from data.transforms.vision_3d.random_plane_crop import RandomPlaneCrop
from data.transforms.vision_3d.random_point_crop import RandomPointCrop


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
        
        super().__init__(data_root=data_root, **kwargs)
        
        # Calculate pairs per source file after annotations are initialized
        self._calculate_pairs_per_file()
    
    def _init_annotations(self) -> None:
        """Initialize source annotations - to be implemented by subclasses.
        
        Subclasses should:
        1. Set self.source_annotations to list of source file annotations
        2. Call super()._calculate_pairs_per_file() if needed
        """
        raise NotImplementedError("Subclasses must implement _init_annotations and set self.source_annotations")
    
    def _calculate_pairs_per_file(self) -> None:
        """Calculate how many synthetic pairs to generate per source file."""
        num_source_files = len(self.source_annotations)
        base_pairs_per_file = self.total_dataset_size // num_source_files
        remainder = self.total_dataset_size % num_source_files
        
        # Distribute pairs evenly with remainder distributed to first files
        self.pairs_per_file = []
        for i in range(num_source_files):
            extra_pair = 1 if i < remainder else 0
            self.pairs_per_file.append(base_pairs_per_file + extra_pair)
        
        # Create flat annotations mapping each datapoint to (file_idx, pair_idx)
        self.annotations = []
        for file_idx, num_pairs in enumerate(self.pairs_per_file):
            for pair_idx in range(num_pairs):
                self.annotations.append({
                    'file_idx': file_idx,
                    'pair_idx': pair_idx,
                    'source_annotation': self.source_annotations[file_idx]
                })
    
    def _load_transform_cache(self) -> None:
        """Load cached transform-to-overlap mappings."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.transform_cache = json.load(f)
        else:
            self.transform_cache = {}
    
    def _save_transform_cache(self) -> None:
        """Save transform-to-overlap mappings to cache."""
        if self.cache_transforms:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.transform_cache, f, indent=2)
    
    def _get_file_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return file_hash
    
    def _load_source_data(self, source_annotation: Dict[str, Any]) -> torch.Tensor:
        """Load source point cloud data - to be implemented by subclasses.
        
        Args:
            source_annotation: Annotation for source file/pair
            
        Returns:
            Point cloud positions as tensor
        """
        raise NotImplementedError("Subclasses must implement _load_source_data")
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a synthetic datapoint using the modular pipeline.
        
        Checks transform-to-overlap cache first, then decides to call either _get_pair or _generate_more.
        """
        file_idx, transform_idx = self._get_indices(idx)
        source_annotation = self.source_annotations[file_idx]
        
        # Get cache key for this file
        file_cache_key = self._get_file_cache_key(source_annotation.get('file_path', str(file_idx)))
        
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
            file_idx: Index of source file
            transform_idx: Index of transform for this file
            valid_transforms: List of valid cached transforms for this file
            
        Returns:
            Tuple of (src_pc, tgt_pc, transform_matrix, transform_config)
        """
        source_annotation = self.source_annotations[file_idx]
        
        # Load source point cloud
        original_pc = self._load_source_data(source_annotation)
        
        # Get the specific transform config from cache
        transform_config = valid_transforms[transform_idx]
        
        # Build transform components from cached params
        transform_matrix, crop_transform = self._build_transform(transform_config)
        
        # Apply transform to get point cloud pair
        src_pc, tgt_pc = self._apply_transform(original_pc, transform_matrix, crop_transform, transform_config)
        
        return src_pc, tgt_pc, transform_matrix, transform_config
    
    def _generate_more(self, original_pc: torch.Tensor, file_cache_key: str, needed_count: int) -> List[Dict[str, Any]]:
        """Generate more valid transforms for a file.
        
        Args:
            original_pc: Original point cloud positions
            file_cache_key: Cache key for this file
            needed_count: Number of additional valid transforms needed
            
        Returns:
            List of new valid transform configurations
        """
        cached_transforms = self.transform_cache.get(file_cache_key, [])
        new_valid_transforms = []
        
        base_seed = hash(file_cache_key) % (2**32)
        trial = len(cached_transforms)  # Start from where cache left off
        
        while len(new_valid_transforms) < needed_count and trial < 1000:  # Safety limit
            # Sample transform parameters
            transform_params = self._sample_transform(base_seed + trial)
            
            # Build transform components
            transform_matrix, crop_transform = self._build_transform(transform_params)
            
            # Apply transform and compute overlap
            src_pc, tgt_pc = self._apply_transform(original_pc, transform_matrix, crop_transform, transform_params)
            
            # Compute overlap
            overlap = compute_registration_overlap(
                ref_points=original_pc,
                src_points=src_pc['pos'],
                transform=None,
                positive_radius=self.matching_radius * 2
            )
            
            # Add overlap to config
            transform_params['overlap'] = float(overlap)
            transform_params['trial'] = trial
            
            # Add to cache (all transforms, regardless of overlap)
            cached_transforms.append(transform_params)
            
            # If overlap is in range, add to valid transforms
            if self.overlap_range[0] < overlap <= self.overlap_range[1]:
                new_valid_transforms.append(transform_params)
            
            trial += 1
        
        # Update cache
        if self.cache_transforms:
            self.transform_cache[file_cache_key] = cached_transforms
            self._save_transform_cache()
        
        return new_valid_transforms
    
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
    
    def _apply_transform(self, original_pc: torch.Tensor, transform_matrix: torch.Tensor, 
                        crop_transform: Any, transform_params: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply transform to create source and target point clouds.
        
        Args:
            original_pc: Original point cloud positions
            transform_matrix: 4x4 transformation matrix
            crop_transform: Crop transform object
            transform_params: Transform configuration
            
        Returns:
            Tuple of (src_pc, tgt_pc) dictionaries
        """
        # Apply SE(3) transformation
        transformed_pc = (transform_matrix[:3, :3] @ original_pc.T).T + transform_matrix[:3, 3]
        
        # Apply crop with deterministic generator
        crop_generator = torch.Generator()
        crop_generator.manual_seed(transform_params['seed'] + 1000)
        
        transformed_pc_dict = {'pos': transformed_pc}
        src_pc_dict = crop_transform._call_single(transformed_pc_dict, generator=crop_generator)
        src_pc_pos = src_pc_dict['pos']
        
        # Create point cloud dictionaries
        src_pc = {
            'pos': src_pc_pos,
            'feat': torch.ones((src_pc_pos.shape[0], 1), dtype=torch.float32),
        }
        
        tgt_pc = {
            'pos': original_pc,
            'feat': torch.ones((original_pc.shape[0], 1), dtype=torch.float32),
        }
        
        return src_pc, tgt_pc
