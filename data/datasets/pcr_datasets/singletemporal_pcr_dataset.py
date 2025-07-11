from typing import Tuple, Dict, Any
import os
import glob
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from utils.io.point_cloud import load_point_cloud


class SingletemporalPCRDataset(SyntheticTransformPCRDataset):
    """Single-temporal point cloud registration dataset.
    
    This class handles single-temporal point cloud datasets where the same file
    is used for both source and target, typically for self-registration tasks.
    
    Features:
    - Single file used for both src and tgt (self-registration)
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
        """Initialize single-temporal PCR dataset.
        
        Args:
            data_root: Path to dataset root directory
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
        """Initialize file pair annotations for single-temporal dataset.
        
        For single-temporal, each file pair has same src_file_path and tgt_file_path.
        """
        # Find all point cloud files
        pc_files = sorted(glob.glob(os.path.join(self.data_root, self.file_extension)))
        
        # Create file pair annotations - for single-temporal, src and tgt are the same file
        self.file_pair_annotations = []
        for file_path in pc_files:
            annotation = {
                'src_file_path': file_path,
                'tgt_file_path': file_path,  # Same file for self-registration
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} files for single-temporal PCR dataset")

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
