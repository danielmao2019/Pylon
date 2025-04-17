from typing import List, Dict, Any
import os
import glob
import json
import torch
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset


class RealPCRDataset(BasePCRDataset):

    def __init__(self, gt_transforms_filepath: str, **kwargs) -> None:
        self.gt_transforms_filepath = gt_transforms_filepath
        super(RealPCRDataset, self).__init__(**kwargs)

    def _init_file_pairs(self) -> None:
        """Initialize source and target file path pairs and their transforms.

        For real PCR, we load the pairs and transforms from a JSON file.
        """
        self.filepaths = sorted(glob.glob(os.path.join(self.data_root, "*.las")))
        self.filepath_pairs = [
            (self.filepaths[i], self.filepaths[0]) for i in range(1, len(self.filepaths))
        ]
        with open(self.gt_transforms_filepath, mode='r') as f:
            gt_transforms: List[Dict[str, Any]] = json.load(f)
        assert len(gt_transforms) == len(self.filepaths)
        assert set(os.path.join(self.data_root, t['filepath']) for t in gt_transforms) == set(self.filepaths), \
            f"{set(os.path.join(self.data_root, t['filepath']) for t in gt_transforms)=}, {set(self.filepaths)=}"
        gt_transforms = sorted(gt_transforms, key=lambda x: x['filepath'])
        gt_transforms = [
            torch.tensor(t['transform'], dtype=torch.float32)
            for t in gt_transforms
        ]
        assert torch.equal(gt_transforms[0], torch.eye(4, dtype=torch.float32))
        self.gt_transforms = gt_transforms[1:]
