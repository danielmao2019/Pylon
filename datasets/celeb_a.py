from typing import Tuple, List, Dict, Callable
import os
import torch
import torchvision
from PIL import Image

import datasets
from .base_dataset import BaseDataset


class CelebA(BaseDataset):
    __doc__ = r"""

    Download: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Used in:
        Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (https://arxiv.org/pdf/2010.06808.pdf)
        Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/pdf/2111.10603.pdf)
        FAMO: Fast Adaptive Multitask Optimization (https://arxiv.org/pdf/2306.03792.pdf)
        Towards Impartial Multi-task Learning (https://openreview.net/pdf?id=IMPnRXEWpvr)
        Multi-Task Learning as Multi-Objective Optimization (https://arxiv.org/pdf/1810.04650.pdf)
        Gradient Surgery for Multi-Task Learning (https://arxiv.org/pdf/2001.06782.pdf)
        Heterogeneous Face Attribute Estimation: A Deep Multi-Task Learning Approach (https://arxiv.org/pdf/1706.00906.pdf)
    """

    TOTAL_SIZE = 202599
    SPLIT_OPTIONS = ['train', 'val', 'test']
    TASK_NAMES = ['landmarks', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    def __init__(
        self, data_root: str, split: str, indices: List[int] = None,
        transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(data_root=data_root, split=split, transforms=transforms, indices=indices)

    ####################################################################################################
    ####################################################################################################

    def _init_images_(self, split: str) -> None:
        # input check
        assert type(split) == str, f"{type(split)=}"
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        # initialize
        split_enum = {0: 'train', 1: 'val', 2: 'test'}
        # images
        images_root = os.path.join(self.data_root, "images", "img_align_celeba")
        self.image_filepaths = []
        with open(os.path.join(self.data_root, "list_eval_partition.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines) == self.TOTAL_SIZE
            for idx in range(self.TOTAL_SIZE):
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{line[0]=}, {idx=}"
                filepath = os.path.join(images_root, line[0])
                assert os.path.isfile(filepath)
                if split_enum[int(line[1])] == split:
                    self.image_filepaths.append(filepath)

    def _init_labels_(self, split: str) -> None:
        self._init_landmarks_labels_()
        self._init_attributes_labels_()

    def _init_landmarks_labels_(self):
        with open(os.path.join(self.data_root, "list_landmarks_align_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert len(lines[0].strip().split()) == 10, f"{lines[0].strip().split()=}"
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            self.landmark_labels = []
            for fp in self.image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                landmarks = torch.tensor(list(map(int, line[1:])), dtype=torch.uint8)
                assert landmarks.shape == (10,), f"{landmarks.shape=}"
                self.landmark_labels.append(landmarks)

    def _init_attributes_labels_(self):
        with open(os.path.join(self.data_root, "list_attr_celeba.txt"), mode='r') as f:
            lines = f.readlines()
            assert set(lines[0].strip().split()) == set(self.TASK_NAMES[1:])
            lines = lines[1:]
            assert len(lines) == self.TOTAL_SIZE
            self.attribute_labels = []
            for fp in self.image_filepaths:
                idx = int(os.path.basename(fp).split('.')[0]) - 1
                line = lines[idx].strip().split()
                assert int(line[0].split('.')[0]) == idx + 1, f"{fp=}, {line[0]=}, {idx=}"
                attributes = dict(
                    (name, torch.tensor([1 if val == "1" else 0], dtype=torch.int8))
                    for name, val in zip(self.TASK_NAMES[1:], line[1:])
                )
                self.attribute_labels.append(attributes)

    ####################################################################################################
    ####################################################################################################

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = self.indices[idx] if self.indices is not None else idx
        result = {
            'image': torchvision.transforms.ToTensor()(Image.open(self.image_filepaths[idx])),
            'landmarks': self.landmark_labels[idx],
        }
        result.update(self.attribute_labels[idx])
        # apply transforms
        result = datasets.utils.apply_transforms(transforms=self.transforms, example=result)
        return result
