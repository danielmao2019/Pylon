from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.random_point_crop import RandomPointCrop


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
        """Initialize annotations with OFF file paths for ModelNet40.
        
        This method creates annotations directly without using file pair annotations,
        following the full BaseDataset pattern.
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
        
        # Create annotations directly - for single-temporal, src and tgt are the same file
        self.annotations = []
        for file_path in off_files:
            annotation = {
                'src_filepath': file_path,
                'tgt_filepath': file_path,  # Same file for self-registration
                'category': self.get_category_from_path(file_path),
            }
            self.annotations.append(annotation)
        
        print(f"Found {len(self.annotations)} OFF files for split '{self.split}'")

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'keep_ratio': self.keep_ratio,
        })
        return version_dict

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
    
    def _apply_crop(self, idx: int, pc_data: dict) -> dict:
        """Build and apply RandomPointCrop transform to point cloud data.
        
        Args:
            idx: Index into self.annotations
            pc_data: Point cloud dictionary
            
        Returns:
            Cropped point cloud dictionary
        """
        # Build RandomPointCrop transform using self.keep_ratio (init parameter)
        crop_transform = RandomPointCrop(
            keep_ratio=self.keep_ratio,
            viewpoint=None,  # Random viewpoint
            limit=500.0  # Default limit
        )
        
        # Use deterministic seeding - combine base_seed with idx for variation
        seed = self.base_seed + idx
        
        # Use __call__ method with seed directly (no manual generator creation)
        return crop_transform(pc_data, seed=seed)
