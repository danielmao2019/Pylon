from typing import Tuple, Dict, Any, List
import os
import glob
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from utils.io.point_cloud import load_point_cloud


class BitemporalPCRDataset(SyntheticTransformPCRDataset):
    """Bi-temporal point cloud registration dataset for multi-scene datasets.
    
    This class handles real point cloud pairs from different files/times,
    using the transform-to-overlap cache system for synthetic pair generation.
    
    Features:
    - Multi-scene point cloud pairs (different files for src/tgt)
    - Transform-to-overlap cache system for synthetic pair generation
    - Parallel processing with deterministic seeding
    - No defensive programming - fail fast with clear errors
    """
    
    # Required BaseDataset attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = None  # Will be set dynamically based on dataset_size parameter
    INPUT_NAMES = ['src_pc', 'tgt_pc', 'correspondences']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(
        self,
        data_root: str,
        dataset_size: int,
        overlap_range: Tuple[float, float] = (0.3, 1.0),
        matching_radius: float = 0.05,
        file_extension: str = "*.las",
        **kwargs,
    ) -> None:
        """Initialize bi-temporal PCR dataset.
        
        Args:
            data_root: Path to dataset root directory with point cloud files
            dataset_size: Total number of synthetic pairs to generate
            overlap_range: Overlap range (overlap_min, overlap_max] for filtering
            matching_radius: Radius for correspondence finding
            file_extension: File extension pattern for point cloud files
            **kwargs: Additional arguments passed to SyntheticTransformPCRDataset
        """
        self.file_extension = file_extension
        
        super().__init__(
            data_root=data_root,
            dataset_size=dataset_size,
            overlap_range=overlap_range,
            matching_radius=matching_radius,
            **kwargs
        )
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations for bi-temporal dataset.
        
        For bi-temporal, each file pair has different src_file_path and tgt_file_path.
        This is a base implementation - subclasses should override for specific datasets.
        """
        # Find all point cloud files
        pc_files = sorted(glob.glob(os.path.join(self.data_root, self.file_extension)))
        
        # Create simple adjacent pairs for demonstration
        # Real datasets would have more sophisticated pairing logic (metadata, timestamps, etc.)
        self.file_pair_annotations = []
        
        for i in range(len(pc_files) - 1):
            annotation = {
                'src_file_path': pc_files[i],
                'tgt_file_path': pc_files[i + 1],  # Different file for bi-temporal
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} file pairs for bi-temporal PCR dataset")

    def _load_single_file(self, file_path: str) -> torch.Tensor:
        """Load point cloud data from a single file.
        
        Args:
            file_path: Path to point cloud file
            
        Returns:
            Point cloud positions as tensor
        """
        # Load raw point cloud
        pc_data = load_point_cloud(file_path)
        
        # Extract position data - let it crash if structure is unexpected
        pc_positions = pc_data['pos'].float() if isinstance(pc_data, dict) else pc_data[:, :3].float()
        
        return pc_positions