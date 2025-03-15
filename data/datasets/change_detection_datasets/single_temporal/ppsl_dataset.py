from typing import Tuple, Dict, Any
import os
import random
import torch
import torchvision
import matplotlib.pyplot as plt
from data.datasets import BaseSyntheticDataset
from utils.input_checks.str_types import check_write_dir


class PPSLDataset(BaseSyntheticDataset):
    __doc__ = r"""
    References:
        * https://github.com/SGao1997/PPSL_MGFDNet/blob/main/dataset_half_bz24.py#L19
    """

    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']

    def __init__(self, **kwargs) -> None:
        super(PPSLDataset, self).__init__(**kwargs)
        self.colorjit = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
        self.affine = torchvision.transforms.RandomAffine(degrees=(-5, 5), scale=(1, 1.02), translate=(0.02, 0.02), shear=(-5, 5))

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        # Fetch the primary datapoint
        img_1 = self.source[idx]['inputs']['image']
        label_1 = self.source[idx]['labels']['semantic_map']

        # Apply color jitter to the first image
        img_1 = self.colorjit(img_1)

        # Select a random second datapoint
        idx_2 = random.choice(range(len(self.source)))
        img_2 = self.source[idx_2]['inputs']['image']
        label_2 = self.source[idx_2]['labels']['semantic_map']

        # Apply affine transformation to the second image
        img_2 = self.affine(img_2)

        # Apply the patch to the second image and label
        img_2_patched = img_2.clone()
        label_2_patched = label_2.clone()
        img_2_patched[:, 256:512, :] = img_1[:, 256:512, :]
        label_2_patched[256:512, :] = label_1[256:512, :]

        # Calculate the change map (logical XOR of labels)
        change_map = (label_1 != label_2_patched).type(torch.int64)

        # Prepare the inputs and labels
        inputs = {
            'img_1': img_1,
            'img_2': img_2_patched,
        }
        labels = {
            'change_map': change_map,
            'semantic_map': label_1,
        }
        meta_info = {
            'semantic_map_1': label_1,
            'semantic_map_2': label_2,
        }

        return inputs, labels, meta_info
