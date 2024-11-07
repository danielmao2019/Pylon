from typing import Tuple, Dict, Any
import random
import torch
import torchvision
from PIL import Image
from data.datasets import BaseDataset
from utils.io import _pil2torch


class MultiMNISTDataset(BaseDataset):

    SPLIT_OPTIONS = ['train', 'val']
    DATASET_SIZE = {
        'train': 60000,
        'val': 10000,
    }
    INPUT_NAMES = ['image']
    LABEL_NAMES = ['left', 'right']
    SHA1SUM = None
    NUM_CLASSES = 10

    def _init_annotations_(self, split: str) -> None:
        self.annotations = torchvision.datasets.MNIST(
            root=self.data_root, train=(split=='train'), download=True,
        )
        return

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        random.seed(idx)
        indices = random.sample(population=range(len(self.annotations)), k=2)
        l_dp = self.annotations[indices[0]]
        r_dp = self.annotations[indices[1]]
        inputs = {
            'image': self._get_image(l_dp[0], r_dp[0]),
        }
        labels = {
            'left': torch.tensor(l_dp[1], dtype=torch.int64),
            'right': torch.tensor(r_dp[1], dtype=torch.int64),
        }
        meta_info = {
            'image_resolution': inputs['image'].shape,
        }
        return inputs, labels, meta_info

    def _get_image(self, l_image: Image.Image, r_image: Image.Image) -> torch.Tensor:
        l_image = _pil2torch(l_image)
        r_image = _pil2torch(r_image)
        assert l_image.ndim == r_image.ndim == 2, f"{l_image.shape=}, {r_image.shape=}"
        assert l_image.shape == r_image.shape, f"{l_image.shape=}, {r_image.shape=}"
        left = torch.cat([l_image, torch.zeros(
            size=(r_image.shape[0], l_image.shape[1]), dtype=torch.float32, device=l_image.device,
        )], dim=0)
        right = torch.cat([torch.zeros(
            size=(l_image.shape[0], r_image.shape[1]), dtype=torch.float32, device=r_image.device,
        ), r_image], dim=0)
        image = torch.cat([left, right], dim=1)
        image = image.unsqueeze(0)
        assert image.shape == (1, l_image.shape[0]+r_image.shape[0], l_image.shape[1]+r_image.shape[1]), f"{image.shape=}"
        return image
