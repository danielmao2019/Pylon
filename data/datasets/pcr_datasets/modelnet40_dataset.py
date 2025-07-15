from typing import Tuple
import os
import glob
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset


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
        data_root: str = '/data/datasets/soft_links/ModelNet40',
        dataset_size: int = 1000,
        overlap_range: Tuple[float, float] = (0.0, 1.0),
        matching_radius: float = 0.05,
        rotation_mag: float = 45.0,
        translation_mag: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialize ModelNet40 dataset.
        
        Args:
            data_root: Path to ModelNet40 dataset root directory
            dataset_size: Total number of synthetic registration pairs to generate
            overlap_range: Overlap range (overlap_min, overlap_max] for generated pairs
            matching_radius: Radius for correspondence finding
            rotation_mag: Maximum rotation magnitude in degrees for synthetic transforms
            translation_mag: Maximum translation magnitude for synthetic transforms
            **kwargs: Additional arguments passed to SyntheticTransformPCRDataset
        """
        super().__init__(
            data_root=data_root,
            dataset_size=dataset_size,
            overlap_range=overlap_range,
            matching_radius=matching_radius,
            rotation_mag=rotation_mag,
            translation_mag=translation_mag,
            **kwargs
        )

    def _init_annotations(self) -> None:
        """Initialize file pair annotations with OFF file paths.
        
        For ModelNet40 (single-temporal), each file pair has same src_file_path and tgt_file_path.
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
                'src_file_path': file_path,
                'tgt_file_path': file_path,  # Same file for self-registration
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
    
    def _normalize_point_cloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Normalize ModelNet40 point cloud to unit sphere following GeoTransformer.
        
        This ensures ModelNet40 objects work with GeoTransformer's standard parameters
        (trans_mag=0.5, matching_radius=0.05, etc.) regardless of original object scale.
        
        Args:
            pc: Point cloud tensor of shape (N, 3)
            
        Returns:
            Normalized point cloud tensor centered at origin with max distance = 1.0
        """
        # Center at origin
        pc_centered = pc - pc.mean(dim=0, keepdim=True)
        
        # Scale to unit sphere (max distance from origin = 1.0)
        max_dist = torch.norm(pc_centered, dim=1).max()
        if max_dist > 0:
            pc_normalized = pc_centered / max_dist
        else:
            pc_normalized = pc_centered
            
        return pc_normalized
