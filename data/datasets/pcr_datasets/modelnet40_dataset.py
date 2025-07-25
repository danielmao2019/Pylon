from typing import Tuple, Dict, Any, List
import os
import glob
import torch
import numpy as np
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.random_point_crop import RandomPointCrop
from utils.point_cloud_ops import normalize_point_cloud


class ModelNet40Dataset(SyntheticTransformPCRDataset):
    """ModelNet40 dataset for point cloud registration.
    
    This dataset implements self-registration on ModelNet40 objects:
    1. Load raw OFF files from ModelNet40
    2. Apply random SE(3) transformations 
    3. Apply random cropping (plane-based or point-based)
    4. Create source/target registration pairs from same object
    """
    
    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'test']  # ModelNet40 only has train/test splits
    DATASET_SIZE = None  # Will be set dynamically based on actual files found
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None  # ModelNet40 doesn't have official checksum
    
    # ModelNet40 object categories
    CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox'
    ]
    
    # Asymmetric categories (have distinct orientations)
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'car', 'chair', 'curtain', 'desk',
        'door', 'dresser', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]

    def __init__(
        self,
        data_root: str,
        dataset_size: int,
        keep_ratio: float = 0.7,
        **kwargs,
    ) -> None:
        """Initialize ModelNet40 dataset with RandomPointCrop.
        
        Args:
            data_root: Path to ModelNet40 root directory
            dataset_size: Total number of synthetic pairs to generate
            keep_ratio: Fraction of points to keep after cropping (0.0 to 1.0)
            **kwargs: Additional arguments passed to parent class
        """
        self.keep_ratio = keep_ratio
        super().__init__(
            data_root=data_root,
            dataset_size=dataset_size,
            **kwargs,
        )
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations with OFF file paths.
        
        For ModelNet40 (single-temporal), each file pair has same src_filepath and tgt_filepath.
        """
        
        # ModelNet40 structure: ModelNet40/[category]/[train|test]/[filename].off
        split_dir = self.split
        if self.split == 'val':
            # Map val to test for ModelNet40 (only has train/test)
            split_dir = 'test'
        
        off_files = []
        
        for category in self.CATEGORIES:
            category_dir = os.path.join(self.data_root, category, split_dir)
            
            if not os.path.exists(category_dir):
                continue
            
            # Find all OFF files in this category/split
            category_files = sorted(glob.glob(os.path.join(category_dir, '*.off')))
            off_files.extend(category_files)
        
        # Create file pair annotations - for single-temporal, src and tgt are the same file
        self.file_pair_annotations = []
        for file_path in off_files:
            annotation = {
                'src_filepath': file_path,
                'tgt_filepath': file_path,  # Same file for self-registration
                'category': self.get_category_from_path(file_path),
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} OFF files for split '{self.split}'")
    
    def get_category_from_path(self, file_path: str) -> str:
        """Extract category from file path.
        
        Args:
            file_path: Path to OFF file
            
        Returns:
            Category name (e.g., 'airplane', 'chair')
        """
        # Path structure: .../ModelNet40/[category]/[train|test]/[filename].off
        path_parts = file_path.split(os.sep)
        
        # Find ModelNet40 in path and get category
        for i, part in enumerate(path_parts):
            if part == 'ModelNet40' and i + 1 < len(path_parts):
                return path_parts[i + 1]
        
        # Fallback: extract from parent directory
        return os.path.basename(os.path.dirname(os.path.dirname(file_path)))

    def _get_cache_param_key(self) -> tuple:
        """Generate cache parameter key including ModelNet40-specific parameters.
        
        Returns:
            Cache parameter key tuple
        """
        parent_key = super()._get_cache_param_key()
        return parent_key + (self.keep_ratio,)

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Load and normalize ModelNet40 point cloud data.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_dict, tgt_pc_dict) normalized point cloud dictionaries
        """
        # Load point clouds using parent method (returns dictionaries)
        src_pc_dict, tgt_pc_dict = super()._load_file_pair_data(file_pair_annotation)
        
        # Apply ModelNet40-specific normalization to position data
        src_pc_dict['pos'] = normalize_point_cloud(src_pc_dict['pos'])
        tgt_pc_dict['pos'] = normalize_point_cloud(tgt_pc_dict['pos'])
        
        return src_pc_dict, tgt_pc_dict

    def _sample_crop(self, seed: int, file_idx: int) -> dict:
        """Sample crop parameters for ModelNet40 with RandomPointCrop.
        
        Args:
            seed: Random seed for deterministic sampling
            file_idx: Index of file pair (unused for ModelNet40)
            
        Returns:
            Dictionary containing crop parameters
        """
        # Generate crop seed for deterministic cropping
        crop_seed = (seed * 31 + 42) % (2**32)
        
        return {
            'crop_seed': crop_seed,
            'keep_ratio': self.keep_ratio,
        }
    
    def _build_crop(self, crop_params: dict) -> RandomPointCrop:
        """Build RandomPointCrop transform from parameters.
        
        Args:
            crop_params: Crop configuration dictionary
            
        Returns:
            RandomPointCrop transform object
        """
        # Create RandomPointCrop transform
        crop_transform = RandomPointCrop(
            keep_ratio=crop_params['keep_ratio'],
            viewpoint=None,  # Random viewpoint
            limit=500.0  # Default limit
        )
        
        return crop_transform
    
    def _apply_crop(self, crop_transform: RandomPointCrop, pc_data: dict, crop_params: dict) -> dict:
        """Apply RandomPointCrop transform to point cloud data.
        
        Args:
            crop_transform: RandomPointCrop transform object
            pc_data: Point cloud dictionary
            crop_params: Crop configuration parameters
            
        Returns:
            Cropped point cloud dictionary
        """
        # Use crop_seed for deterministic cropping
        crop_seed = crop_params['crop_seed']
        generator = torch.Generator(device=self.device)
        generator.manual_seed(crop_seed)
        
        return crop_transform._call_single(pc_data, generator=generator)
