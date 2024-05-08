from typing import Tuple, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
from ..transforms.compose import Compose
from utils.input_checks import check_read_dir
from utils.builder import build_from_config


class BaseDataset(ABC, torch.utils.data.Dataset):

    SPLIT_OPTIONS: List[str] = None
    INPUT_NAMES: List[str] = None
    LABEL_NAMES: List[str] = None

    def __init__(
        self,
        data_root: str,
        split: str,
        transforms: Optional[list] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        super(BaseDataset, self).__init__()
        # sanity checks
        assert self.SPLIT_OPTIONS is not None
        assert self.INPUT_NAMES is not None
        assert self.LABEL_NAMES is not None
        assert set(self.INPUT_NAMES) & set(self.LABEL_NAMES) == set(), f"{set(self.INPUT_NAMES) & set(self.LABEL_NAMES)=}"
        # initialize data root
        self.data_root = check_read_dir(data_root)
        # initialize annotations
        assert type(split) == str, f"{type(split)=}"
        if split in self.SPLIT_OPTIONS:
            self._init_annotations_(split=split)
            if indices is not None:
                self.annotations = [self.annotations[idx] for idx in indices]
        else:
            self.annotations = []
        # initialize transform
        self._init_transform_(transforms=transforms)

    @abstractmethod
    def _init_annotations_(self, split: str) -> None:
        raise NotImplementedError("[ERROR] _init_annotations_ not implemented for abstract base class.")

    def _init_transform_(self, transforms: Optional[list]):
        if transforms is None:
            transforms = {
                'class': Compose,
                'args': {
                    'transforms': [],
                },
            }
        self.transforms = build_from_config(transforms)

    def __len__(self):
        return len(self.annotations)

    @abstractmethod
    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        r"""This method defines how inputs, labels, and meta info are loaded from disk.

        Args:
            idx (int): index of data point.

        Returns:
            inputs (Dict[str, torch.Tensor]): the inputs to the model.
            labels (Dict[str, torch.Tensor]): the ground truth for the current inputs.
            meta_info (Dict[str, Any]): the meta info for the current data point.
        """
        raise NotImplementedError("[ERROR] _load_example_ not implemented for abstract base class.")

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Any]]:
        inputs, labels, meta_info = self._load_example_(idx)
        example = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info,
        }
        example = self.transforms(example)
        return example
