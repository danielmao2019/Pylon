from typing import List, Dict, Any, Optional
from abc import ABC
import torch
from data.datasets import BaseDataset


class BaseSyntheticDataset(BaseDataset, ABC):

    def __init__(
        self,
        source: BaseDataset,
        dataset_size: int,
        transforms_cfg: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        assert isinstance(source, BaseDataset)
        self.source = source
        assert isinstance(dataset_size, int) and dataset_size >= 0
        self.DATASET_SIZE = dataset_size
        self._init_transforms(transforms_cfg=transforms_cfg)
        if use_cache:
            self.cache: List[Dict[str, Dict[str, Any]]] = []
        else:
            self.cache = None
        self.device = device

    def __len__(self) -> int:
        return self.DATASET_SIZE
