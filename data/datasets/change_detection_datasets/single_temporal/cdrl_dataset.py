from typing import Tuple, Dict, Callable, Any
import torch
from data.datasets import BaseSyntheticDataset


class CDRLDataset(BaseSyntheticDataset):
    __doc__ = r"""
    Using CycleGAN as transform:
        * https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        * [latest_net_G_A](https://drive.google.com/file/d/1M7fIJo6koqLFqXVjKG0PWHRWlTPN5BZV/view?usp=sharing)
        * [latest_net_G_B](https://drive.google.com/file/d/1k_tGVaI-4_Wn6-eLT0qvm8YsIz9oDqnS/view?usp=sharing)
    """

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor], **kwargs) -> None:
        assert callable(transform)
        self.transform = transform
        super(CDRLDataset, self).__init__(**kwargs)

    def _load_datapoint(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        datapoint = self.dataset[idx]
        img_1 = datapoint['inputs']['image']
        img_2 = self.transform(img_1)
        inputs = {
            'img_1': img_1,
            'img_2': img_2,
        }
        labels = {
            'target': img_1,
        }
        meta_info = {}
        return inputs, labels, meta_info
