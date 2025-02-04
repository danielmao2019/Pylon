from typing import Tuple, Dict, Any
import torch
import torchvision
from data.datasets import BaseSyntheticDataset


class MNISTDataset(BaseSyntheticDataset):

    SPLIT_OPTIONS = ['train', 'test']

    def __init__(self, split: str, **kwargs):
        assert split in self.SPLIT_OPTIONS, f"{split=}, {self.SPLIT_OPTIONS=}"
        self.split = split
        assert {'source', 'dataset_size'} & set(kwargs.keys()) == set()
        super(MNISTDataset, self).__init__(
            source=torchvision.datasets.MNIST(
                root=self.data_root,
                train=self.split=='train',
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),  # Convert to tensor
                ])
            ), **kwargs)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        image, label = self.mnist[idx]

        inputs = {
            'image': image.to(self.device),  # Ensure it's on the right device
        }
        labels = {
            'label': torch.tensor(label, dtype=torch.int64, device=self.device),
        }
        meta_info = {
            'index': idx,
        }
        return inputs, labels, meta_info
