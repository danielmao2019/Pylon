from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
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

    def is_asymmetric_object(self, file_path: str) -> bool:
        """Check if object belongs to asymmetric category.
        
        Args:
            file_path: Path to OFF file
            
        Returns:
            True if object is asymmetric, False otherwise
        """
        category = self.get_category_from_path(file_path)
        return category in self.ASYMMETRIC_CATEGORIES

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and normalize ModelNet40 point cloud data.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_normalized, tgt_pc_normalized) point cloud position tensors
        """
        # Load point clouds using parent method
        src_pc_raw, tgt_pc_raw = super()._load_file_pair_data(file_pair_annotation)
        
        # Apply ModelNet40-specific normalization
        src_pc_normalized = normalize_point_cloud(src_pc_raw)
        tgt_pc_normalized = normalize_point_cloud(tgt_pc_raw)
        
        return src_pc_normalized, tgt_pc_normalized
