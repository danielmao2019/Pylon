from typing import Tuple, List, Dict, Any
import os
import torch
import torchvision
from PIL import Image

from .base_dataset import BaseDataset


class MultiTaskFacialLandmarkDataset(BaseDataset):
    __doc__ = r"""

    Download:
        https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
    
    Used in:
        Facial Landmark Detection by Deep Multi-task Learning (https://link.springer.com/chapter/10.1007/978-3-319-10599-4_7)
    """

    SPLIT_OPTIONS = ['train', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['landmarks', 'gender', 'smile', 'glasses', 'pose']

    def __init__(
        self, data_root: str, split: str, indices: List[int] = None,
        transforms: dict = None,
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
        self.labels: List[Dict[str, torch.Tensor]] = []
        with open(os.path.join(self.data_root, f"{split}ing.txt"), mode='r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split()
                assert line[0] == os.path.relpath(path=self.image_filepaths[idx], start=self.data_root), \
                    f"{idx=}, {line[0]=}, {self.image_filepaths[idx]=}, {self.data_root=}"
                landmarks = torch.tensor(list(map(float, [c for coord in zip(line[1:6], line[6:11]) for c in coord])), dtype=torch.float32)
                attributes = dict(
                    (name, torch.tensor(int(val), dtype=torch.int8))
                    for name, val in zip(self.LABEL_NAMES[1:], line[11:15])
                )
                labels: Dict[str, torch.Tensor] = {}
                labels.update({'landmarks': landmarks})
                labels.update(attributes)
                self.labels.append(labels)

    ####################################################################################################
    ####################################################################################################

    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        inputs = {'image': torchvision.transforms.ToTensor()(Image.open(self.image_filepaths[idx]))}
        labels = self.labels[idx]
        meta_info = {
            'image_filepath': os.path.relpath(path=self.image_filepaths[idx], start=self.data_root),
            'image_resolution': tuple(inputs['image'].shape[-2:]),
        }
        return inputs, labels, meta_info
