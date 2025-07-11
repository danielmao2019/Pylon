from typing import Tuple, Dict, Any, List
import os
import glob
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


class BitemporalPCRDataset(BasePCRDataset):
    """Bi-temporal point cloud registration dataset for multi-scene datasets.
    
    This class handles real point cloud pairs with ground truth transformations,
    voxel-based sampling, and complex preprocessing pipelines.
    
    Features:
    - Multi-scene point cloud pairs (different files for src/tgt)
    - Voxel-based grid sampling
    - Complex caching and preprocessing
    - Overlap control through spatial shifts
    """
    
    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': None, 'val': None, 'test': None}
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        data_root: str,
        voxel_size: float = 50.0,
        min_points: int = 256,
        max_points: int = 8192,
        matching_radius: float = 0.1,
        overlap: float = 1.0,
        cache_dirname: str = None,
        **kwargs,
    ) -> None:
        """Initialize bi-temporal PCR dataset.
        
        Args:
            data_root: Path to dataset root directory with .las files
            voxel_size: Size of voxel cells for sampling
            min_points: Minimum number of points in a cluster
            max_points: Maximum number of points in a cluster
            matching_radius: Radius for finding correspondences
            overlap: Desired overlap ratio between 0 and 1
            cache_dirname: Name for cache directory
            **kwargs: Additional arguments passed to BasePCRDataset
        """
        super().__init__(
            data_root=data_root,
            voxel_size=voxel_size,
            min_points=min_points,
            max_points=max_points,
            matching_radius=matching_radius,
            overlap=overlap,
            cache_dirname=cache_dirname or 'bitemporal_pcr_cache',
            **kwargs
        )
    
    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs for bi-temporal PCR datasets.
        
        For bi-temporal datasets, this typically involves:
        - Loading file pairs from metadata
        - Reading ground truth transformations
        - Setting up temporal or spatial correspondence
        
        This is a base implementation - subclasses should override for specific datasets.
        """
        # Get all .las files
        las_files = sorted(glob.glob(os.path.join(self.data_root, '*.las')))
        
        # Create simple adjacent pairs for demonstration
        # Real datasets would have more sophisticated pairing logic
        self.src_file_paths = []
        self.tgt_file_paths = []
        self.gt_transforms = []
        
        for i in range(len(las_files) - 1):
            self.src_file_paths.append(las_files[i])
            self.tgt_file_paths.append(las_files[i + 1])
            
            # Identity transform as placeholder - real datasets would load actual transforms
            self.gt_transforms.append(torch.eye(4, dtype=torch.float32))
        
        # Create filepath pairs for BasePCRDataset compatibility
        self.filepath_pairs = list(zip(self.src_file_paths, self.tgt_file_paths))
        
        print(f"Initialized {len(self.filepath_pairs)} file pairs for bi-temporal PCR dataset")