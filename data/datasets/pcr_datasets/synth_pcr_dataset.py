import os
import glob
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


class SynthPCRDataset(BasePCRDataset):

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.
        
        For synthetic PCR, we use the same file paths for both source and target.
        """
        self.filepaths = sorted(glob.glob(os.path.join(self.data_root, "*.las")))
        self.filepath_pairs = list(zip(self.filepaths, self.filepaths))
        self.transforms = [torch.eye(4, dtype=torch.float32, device=self.device) for _ in self.filepaths]
