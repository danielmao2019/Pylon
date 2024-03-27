from typing import Tuple, List, Dict, Callable
import os
import torch
import torchvision
from PIL import Image

import datasets
from .base_dataset import BaseDataset


class MultiTaskFacialLandmark(BaseDataset):
    __doc__ = r"""

    Download:
        https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
    
    Used in:
        Facial Landmark Detection by Deep Multi-task Learning (https://link.springer.com/chapter/10.1007/978-3-319-10599-4_7)
    """

    SPLIT_OPTIONS = ['train', 'test']
    TASK_NAMES = ['landmarks', 'gender', 'smile', 'glasses', 'pose']

    def __init__(
        self, data_root: str, split: str, indices: List[int] = None,
        transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(data_root=data_root, split=split, transforms=transforms, indices=indices)

    ####################################################################################################
    ####################################################################################################

    def _init_images_(self, split: str) -> None:
        self.image_filepaths = []
        with open(os.path.join(self.data_root, f"{split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                filepath = os.path.join(self.data_root, line[0])
                assert os.path.isfile(filepath), f"{filepath=}"
                self.image_filepaths.append(filepath)

    def _init_labels_(self, split: str) -> None:
        # input check
        assert type(split) == str, f"{type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        # image
        self.labels = []
        with open(os.path.join(self.data_root, f"{split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split()
                assert line[0] == os.path.relpath(path=self.image_filepaths[idx], start=self.data_root), \
                    f"{idx=}, {line[0]=}, {self.image_filepaths[idx]=}, {self.data_root=}"
                landmarks = torch.tensor(list(map(float, [c for coord in zip(line[1:6], line[6:11]) for c in coord])), dtype=torch.float32)
                attributes = dict(
                    (name, torch.tensor(int(val), dtype=torch.int8))
                    for name, val in zip(self.TASK_NAMES[1:], line[11:15])
                )
                labels = {}
                labels.update({'landmarks': landmarks})
                labels.update(attributes)
                self.labels.append(labels)

    ####################################################################################################
    ####################################################################################################

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = self.indices[idx] if self.indices is not None else idx
        result = {'image': torchvision.transforms.ToTensor()(Image.open(self.image_filepaths[idx]))}
        result.update(self.labels[idx])
        # apply transforms
        result = datasets.utils.apply_transforms(transforms=self.transforms, example=result)
        return result
