from typing import Tuple, Dict, Callable, Any
import torch
from data.datasets import BaseSyntheticDataset


class CDRLDataset(BaseSyntheticDataset):

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
