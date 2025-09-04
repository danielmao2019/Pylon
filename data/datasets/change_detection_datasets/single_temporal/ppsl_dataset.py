from typing import Tuple, Dict, Any, Optional, List
import random
import torch
import torchvision
from data.datasets import BaseSyntheticDataset
from data.transforms.torchvision_wrapper import TorchvisionWrapper


class PPSLDataset(BaseSyntheticDataset):
    __doc__ = r"""
    References:
        * https://github.com/SGao1997/PPSL_MGFDNet/blob/main/dataset_half_bz24.py#L19
    """

    INPUT_NAMES = ['img_1', 'img_2']
    LABEL_NAMES = ['change_map']

    def __init__(self, **kwargs) -> None:
        super(PPSLDataset, self).__init__(**kwargs)
        self.colorjit = TorchvisionWrapper(torchvision.transforms.ColorJitter, brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)
        self.affine = TorchvisionWrapper(torchvision.transforms.RandomAffine, degrees=(-5, 5), scale=(1, 1.02), translate=(0.02, 0.02), shear=(-5, 5))

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        version_dict = super()._get_cache_version_dict()
        # PPSLDataset uses ColorJitter and RandomAffine transforms
        # Note: The exact transform parameters are hardcoded in __init__
        version_dict.update({
            'colorjit_brightness': 0.7,
            'colorjit_contrast': 0.7,
            'colorjit_saturation': 0.7,
            'colorjit_hue': 0.2,
            'affine_degrees': (-5, 5),
            'affine_scale': (1, 1.02),
            'affine_translate': (0.02, 0.02),
            'affine_shear': (-5, 5),
        })
        return version_dict

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        # Fetch the primary datapoint
        img_1 = self.source[idx]['inputs']['image']
        label_1 = self.source[idx]['labels']['semantic_map']

        # Apply color jitter to the first image with deterministic seeding
        img_1 = self.colorjit(img_1, seed=(self.base_seed or 0) + idx)

        # Select a deterministic second datapoint using proper pseudo-random generation
        rng = random.Random((self.base_seed or 0) + idx + 500)  # +500 to avoid collision with transforms
        idx_2 = rng.choice(range(len(self.source)))
        img_2 = self.source[idx_2]['inputs']['image']
        label_2 = self.source[idx_2]['labels']['semantic_map']

        # Apply affine transformation to the second image with deterministic seeding
        img_2 = self.affine(img_2, seed=(self.base_seed or 0) + idx + 1000)

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
    
    def display_datapoint(
        self,
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> None:
        """Minimal display_datapoint implementation for synthetic datasets.
        
        Full visualization support for synthetic datasets is not yet implemented.
        """
        return None
