from typing import Tuple, Dict, Any
from data.datasets import BaseSyntheticDataset
import random
import torch
import torchvision


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

        # Create the patch from the top half of the first image and label
        patch_img = img_1[:, 256:512, :]  # Extract patch from the first image
        patch_label = label_1[256:512, :]  # Extract patch from the first label

        # Apply the patch to the second image and label
        img_2_patched = img_2.clone()  # Ensure we don't modify the original tensor
        label_2_patched = label_2.clone()

        img_2_patched[:, 256:512, :] = patch_img
        label_2_patched[256:512, :] = patch_label

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
        meta_info = {}

        return inputs, labels, meta_info
