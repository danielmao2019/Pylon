from typing import List
import os
import glob
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset


class ModelNetDataset(SyntheticTransformPCRDataset):
    """ModelNet40 dataset for point cloud registration.
    
    This dataset implements self-registration on ModelNet40 objects following
    GeoTransformer's approach:
    1. Load raw OFF files from ModelNet40
    2. Generate synthetic transforms with trial-and-error for desired overlaps
    3. Apply random cropping (plane-based or point-based) to create realistic pairs
    4. Cache config-to-overlap mappings for efficiency
    
    The dataset supports both symmetric and asymmetric object categories.
    """
    
    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'test']  # ModelNet40 only has train/test splits
    DATASET_SIZE = {'train': 9843, 'test': 2468}
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
        overlap_range: tuple = (0.3, 1.0),
        matching_radius: float = 0.05,
        max_trials: int = 100,
        config_cache_dir: str = None,
        **kwargs,
    ) -> None:
        """Initialize ModelNet40 dataset.
        
        Args:
            data_root: Path to ModelNet40 dataset root directory
            overlap_range: Overlap range (overlap_min, overlap_max] - left exclusive, right inclusive
            matching_radius: Radius for correspondence finding and overlap computation
            max_trials: Maximum number of trials for finding valid overlap
            config_cache_dir: Directory to cache config-to-overlap mappings
            **kwargs: Additional arguments passed to SyntheticTransformPCRDataset
        """
        assert isinstance(data_root, str), f"data_root must be str, got {type(data_root)}"
        assert os.path.exists(data_root), f"data_root does not exist: {data_root}"
        
        # Set default cache directory to be in data_root
        if config_cache_dir is None:
            split_name = kwargs.get('split', 'unknown')
            overlap_str = f"{overlap_range[0]:.1f}_{overlap_range[1]:.1f}"
            config_cache_dir = os.path.join(data_root, f'modelnet_config_cache_{split_name}_{overlap_str}')
        
        super().__init__(
            data_root=data_root,
            overlap_range=overlap_range,
            matching_radius=matching_radius,
            max_trials=max_trials,
            config_cache_dir=config_cache_dir,
            **kwargs
        )

    def _init_file_paths(self) -> None:
        """Initialize raw file paths for ModelNet40 OFF files.
        
        This method sets up self.raw_file_paths containing paths to all OFF files
        for the current split (train/test).
        """
        assert hasattr(self, 'split'), "Split must be set before initializing file paths"
        assert self.split in self.SPLIT_OPTIONS, f"Invalid split: {self.split}, must be one of {self.SPLIT_OPTIONS}"
        
        # ModelNet40 structure: ModelNet40/[category]/[train|test]/[filename].off
        split_dir = self.split
        if self.split == 'val':
            # Map val to test for ModelNet40 (only has train/test)
            split_dir = 'test'
        
        raw_file_paths = []
        
        for category in self.CATEGORIES:
            category_dir = os.path.join(self.data_root, category, split_dir)
            
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            # Find all OFF files in this category/split
            off_files = sorted(glob.glob(os.path.join(category_dir, '*.off')))
            raw_file_paths.extend(off_files)
        
        self.raw_file_paths = raw_file_paths
        
        if len(self.raw_file_paths) == 0:
            raise ValueError(f"No OFF files found in {self.data_root} for split '{self.split}'")
        
        print(f"Found {len(self.raw_file_paths)} OFF files for split '{self.split}'")

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

