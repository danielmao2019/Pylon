from typing import Tuple, Dict, Any
from data.datasets import BaseSyntheticDataset
import random
import torch


class PPSLDataset(BaseSyntheticDataset):

    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        # Fetch the primary datapoint
        img_1 = self.dataset[idx]['inputs']['image']
        label_1 = self.dataset[idx]['labels']['semantic_segmentation']

        # Select a random second datapoint
        idx_2 = random.choice(range(len(self.dataset)))
        img_2 = self.dataset[idx_2]['inputs']['image']
        label_2 = self.dataset[idx_2]['labels']['semantic_segmentation']

        # Create the patch from the top half of the first image and label
        patch_img = img_1[:, 256:512, :]
        patch_label = label_1[256:512, :]

        # Apply the patch to the second image and label
        img_2_patched = img_2.clone()  # Ensure we don't modify the original tensor
        label_2_patched = label_2.clone()

        img_2_patched[:, 256:512, :] = patch_img
        label_2_patched[256:512, :] = patch_label

        # Calculate the change label
        change_map = (label_1 != label_2_patched).type(torch.int64)

        inputs = {
            'img_1': img_1,
            'img_2': img_2_patched,
        }
        labels = {
            'change_map': change_map,
        }
        meta_info = {}
        return inputs, labels, meta_info
