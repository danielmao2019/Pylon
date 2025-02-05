from typing import List, Dict, Any, Optional
from abc import ABC
import torch
from data.datasets import BaseDataset


class BaseSyntheticDataset(BaseDataset, ABC):

    def __init__(
        self,
        source: BaseDataset,
        dataset_size: Optional[int],
        transforms_cfg: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        device: Optional[torch.device] = torch.device('cuda'),
    ) -> None:
        assert isinstance(source, BaseDataset)
        source.device = torch.device('cpu')
        self.source = source
        self._init_dataset_size(dataset_size)
        self._init_transforms(transforms_cfg=transforms_cfg)
        if use_cache:
            self.cache: List[Dict[str, Dict[str, Any]]] = []
        else:
            self.cache = None
        self._init_device(device)

    def _init_dataset_size(self, dataset_size: Optional[int]) -> None:
        if dataset_size is None:
            dataset_size = len(self.source)
        assert isinstance(dataset_size, int) and dataset_size >= 0
        self.DATASET_SIZE = dataset_size

    def _init_annotations(self) -> None:
        """This abstract method is not needed but must be implemented.
        """
        pass

    def __len__(self) -> int:
        return self.DATASET_SIZE
