import os
import glob
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


class SingletemporalPCRDataset(BasePCRDataset):
    """Single-temporal point cloud registration dataset.
    
    This class handles single-temporal point cloud datasets where the same file
    is used for both source and target, typically for self-registration tasks.
    
    Features:
    - Single file used for both src and tgt (self-registration)
    - Inherits voxel-based grid sampling from BasePCRDataset
    - Complex caching and preprocessing capabilities
    """

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.

        For single-temporal PCR, we use the same file paths for both source and target.
        """
        self.filepaths = sorted(glob.glob(os.path.join(self.data_root, "*.las")))
        self.filepath_pairs = list(zip(self.filepaths, self.filepaths))
        self.gt_transforms = [torch.eye(4, dtype=torch.float32) for _ in self.filepaths]
