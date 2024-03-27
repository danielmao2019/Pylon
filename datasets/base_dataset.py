from typing import List, Dict, Callable
from abc import abstractmethod
import os
import torch


class BaseDataset(torch.utils.data.Dataset):

    SPLIT_OPTIONS: List[str] = None
    TASK_NAMES: List[str] = None
    image_filepaths: List[str] = None

    def __init__(
        self,
        data_root: str,
        split: str,
        transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = None,
        indices: List[int] = None,
    ):
        super(BaseDataset, self).__init__()
        assert self.SPLIT_OPTIONS is not None
        assert self.TASK_NAMES is not None
        # initialize data root directory
        assert type(data_root) == str, f"{type(data_root)=}"
        assert os.path.isdir(data_root), f"{data_root=}"
        self.data_root = data_root
        self.indices = indices
        self._init_images_(split=split)
        self._init_labels_(split=split)
        self._init_transforms_(transforms=transforms)

    @abstractmethod
    def _init_images_(self, split: str) -> None:
        pass

    @abstractmethod
    def _init_labels_(self, split: str) -> None:
        pass

    def _init_transforms_(self, transforms: Dict[str, Callable]) -> None:
        if transforms is None:
            transforms = {}
        assert type(transforms) == dict, f"{type(transforms)=}"
        for key, val in transforms.items():
            assert type(key) == str, f"{type(key)=}"
            assert callable(val), f"{type(val)=}"
        assert set(transforms.keys()).issubset(set(self.TASK_NAMES+['image'])), f"{set(transforms.keys())=}, {set(self.TASK_NAMES+['image'])=}"
        self.transforms = transforms

    def __len__(self):
        if self.indices is None:
            return len(self.image_filepaths)
        else:
            return len(self.indices)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass
