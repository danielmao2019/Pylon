from typing import Tuple, Dict, Any, Optional
import os
import glob
import json
import torch
from data.datasets.pcr_datasets.synthetic_transform_pcr_dataset import SyntheticTransformPCRDataset
from data.transforms.vision_3d.pcr_translation import PCRTranslation


class BiTemporalPCRDataset(SyntheticTransformPCRDataset):
    """Bi-temporal point cloud registration dataset for multi-scene datasets.
    
    This class handles real point cloud pairs from different files/times,
    using the transform-to-overlap cache system for synthetic pair generation.
    
    Features:
    - Multi-scene point cloud pairs (different files for src/tgt)
    - Transform-to-overlap cache system for synthetic pair generation
    - Optional ground truth transforms for unregistered pairs
    - Parallel processing with deterministic seeding
    - No defensive programming - fail fast with clear errors
    """
    
    def __init__(
        self,
        gt_transforms: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize bi-temporal PCR dataset.
        
        Args:
            gt_transforms: Optional path to JSON file with ground truth transforms
                          for pre-registering unregistered point cloud pairs
            **kwargs: Additional arguments passed to SyntheticTransformPCRDataset
                     (including data_root and dataset_size)
        """
        self.gt_transforms_path = gt_transforms
        self.gt_transforms_data = {}
        
        # Load ground truth transforms if provided
        if self.gt_transforms_path is not None:
            assert os.path.exists(self.gt_transforms_path), f"GT transforms file not found: {self.gt_transforms_path}"
            with open(self.gt_transforms_path, 'r') as f:
                self.gt_transforms_data = json.load(f)
                # Expected format: {"src_filepath|tgt_filepath": transform_matrix_list}
        
        super().__init__(**kwargs)
    
    def _init_annotations(self) -> None:
        """Initialize file pair annotations for bi-temporal dataset.
        
        For bi-temporal, each file pair has different src_filepath and tgt_filepath.
        This is a base implementation - subclasses should override for specific datasets.
        """
        # Find all point cloud files (utils/io/point_cloud.py handles all formats)
        all_files = []
        for pattern in ['*.ply', '*.las', '*.laz', '*.txt', '*.pth', '*.off']:
            all_files.extend(glob.glob(os.path.join(self.data_root, pattern)))
        pc_files = sorted(all_files)
        
        # Create simple adjacent pairs for demonstration
        # Real datasets would have more sophisticated pairing logic (metadata, timestamps, etc.)
        self.file_pair_annotations = []
        
        for i in range(len(pc_files) - 1):
            annotation = {
                'src_filepath': pc_files[i],
                'tgt_filepath': pc_files[i + 1],  # Different file for bi-temporal
            }
            self.file_pair_annotations.append(annotation)
        
        print(f"Found {len(self.file_pair_annotations)} file pairs for bi-temporal PCR dataset")
    
    def _load_file_pair_data(self, file_pair_annotation: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point cloud data with optional GT transform pre-registration and PCRTranslation centering.
        
        Args:
            file_pair_annotation: Annotation with 'src_filepath' and 'tgt_filepath' keys
            
        Returns:
            Tuple of (src_pc_raw, tgt_pc_raw) centered point cloud position tensors
        """
        # Load point clouds using parent method
        src_pc_raw, tgt_pc_raw = super()._load_file_pair_data(file_pair_annotation)
        
        # Apply GT transform if available for unregistered data
        if self.gt_transforms_data:
            src_path = file_pair_annotation['src_filepath']
            tgt_path = file_pair_annotation['tgt_filepath']
            transform_key = f"{src_path}|{tgt_path}"
            
            if transform_key in self.gt_transforms_data:
                # Get GT transform matrix
                gt_transform_list = self.gt_transforms_data[transform_key]
                gt_transform = torch.tensor(gt_transform_list, dtype=torch.float32)
                
                # Apply GT transform to register source to target
                from utils.point_cloud_ops import apply_transform
                src_pc_raw = apply_transform(src_pc_raw, gt_transform)
        
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
