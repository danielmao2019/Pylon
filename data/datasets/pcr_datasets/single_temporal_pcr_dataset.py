import os
import glob
import torch
from typing import Tuple, Dict, Any
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.pcr_translation import PCRTranslation


class SingleTemporalPCRDataset(SyntheticTransformPCRDataset):
    """Single-temporal point cloud registration dataset.
    
    This class handles single-temporal point cloud datasets where the same file
    is used for both source and target, typically for self-registration tasks.
    
    Features:
    - Single file used for both src and tgt (self-registration)
    - Transform-to-overlap cache system for synthetic pair generation
    - Parallel processing with deterministic seeding
    - No defensive programming - fail fast with clear errors
    """

    def _init_annotations(self) -> None:
        """Initialize file pair annotations for single-temporal dataset.
        
        For single-temporal, each file pair has same src_filepath and tgt_filepath.
        """
        # Find all point cloud files (utils/io/point_cloud.py handles all formats)
        all_files = []
        for pattern in ['*.ply', '*.las', '*.laz', '*.txt', '*.pth', '*.off']:
            all_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        pc_files = sorted(all_files)
        
        # Create file pair annotations - for single-temporal, src and tgt are the same file
        self.file_pair_annotations = []
        for file_path in pc_files:
            annotation = {
                'src_filepath': file_path,
                'tgt_filepath': file_path,  # Same file for self-registration
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} files for single-temporal PCR dataset")

    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud data with PCRTranslation centering applied.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_raw, tgt_pc_raw) centered point cloud position tensors
        """
        # Load raw point clouds using parent method
        src_pc_raw, tgt_pc_raw = super()._load_file_pair_data(file_pair_annotation)
        
        # Create point cloud dictionaries for PCRTranslation
        src_pc_dict = {'pos': src_pc_raw}
        tgt_pc_dict = {'pos': tgt_pc_raw}
        
        # Create identity transform (PCRTranslation will adjust this appropriately)
        identity_transform = torch.eye(4, dtype=torch.float32, device=self.device)
        
        # Apply PCRTranslation to center both point clouds
        pcr_translation = PCRTranslation()
        centered_src_pc, centered_tgt_pc, _ = pcr_translation(
            src_pc=src_pc_dict,
            tgt_pc=tgt_pc_dict,
            transform=identity_transform
        )
        
        # Return centered position tensors
        return centered_src_pc['pos'], centered_tgt_pc['pos']
