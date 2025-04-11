from typing import List, Dict, Any
import os
import glob
import json
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


class RealPCRDataset(BasePCRDataset):

    def __init__(self, gt_transforms: str, **kwargs) -> None:
        self.gt_transforms = gt_transforms
        super(RealPCRDataset, self).__init__(**kwargs)

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.
        
        For real PCR, we load the pairs and transforms from a JSON file.
        """
        self.filepaths = sorted(glob.glob(os.path.join(self.data_root, "*.las")))
        self.filepath_pairs = [
            (self.filepaths[i], self.filepaths[0]) for i in range(1, len(self.filepaths))
        ]
        with open(self.gt_transforms, mode='r') as f:
            self.gt_transforms: List[Dict[str, Any]] = json.load(f)
        assert len(self.gt_transforms) == len(self.filepaths)
        assert set(os.path.join(self.data_root, t['filepath']) for t in self.gt_transforms) == set(self.filepaths), \
            f"{set(os.path.join(self.data_root, t['filepath']) for t in self.gt_transforms)=}, {set(self.filepaths)=}"
        self.gt_transforms = sorted(self.gt_transforms, key=lambda x: x['filepath'])
        self.transforms = [
            torch.tensor(t['transform'], dtype=torch.float32, device=self.device)
            for t in self.gt_transforms
        ]
        assert torch.equal(self.transforms[0], torch.eye(4, dtype=torch.float32, device=self.device))
        self.transforms = self.transforms[1:]
