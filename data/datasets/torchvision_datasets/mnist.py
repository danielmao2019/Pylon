from typing import Tuple, Dict, Any
import torch
import torchvision
from data.datasets import BaseDataset


class MNISTDataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'test']
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['label']

    def _init_annotations(self) -> None:
        self.annotations = torchvision.datasets.MNIST(
            root=self.data_root,
            train=self.split=='train',
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),  # Convert to tensor
            ])
        )

    def _get_cache_version_dict(self) -> Dict[str, Any]:
        """Return parameters that affect dataset content for cache versioning."""
        # MNISTDataset has no additional parameters beyond BaseDataset
        return super()._get_cache_version_dict()

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        image, label = self.annotations[idx]

        inputs = {
            'image': image,
        }
        labels = {
            'label': torch.tensor(label, dtype=torch.int64),
        }
        meta_info = {
        }
        return inputs, labels, meta_info
