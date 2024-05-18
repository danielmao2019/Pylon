from typing import Dict, Tuple, Any, Optional
import math
import torch
import torchvision
from ..base_dataset import BaseDataset


class SemanticSegmentationOverfitDataset(BaseDataset):

    def __init__(self, num_classes: int, num_examples: int, initial_seed: Optional[int] = None):
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        self.num_classes = num_classes
        assert type(num_examples) == int, f"{type(num_examples)=}"
        self.sample_res = math.ceil(math.sqrt(num_classes))
        assert num_examples >= 0, f"{num_examples=}"
        self.num_examples = num_examples
        self._init_generator_(initial_seed)
        self._init_transform_(transforms=None)

    def _init_annotations_(self, split: str) -> None:
        r"""Intentionally doing nothing.
        """
        pass

    def _init_generator_(self, initial_seed: Optional[int]):
        self.generator = torch.Generator()
        if initial_seed is not None:
            self.generator.manual_seed(initial_seed)
        self.initial_seed = self.generator.initial_seed()

    def __len__(self):
        return self.num_examples

    def _load_example_(self, idx: int) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
    ]:
        seed = self.initial_seed + idx
        self.generator.manual_seed(seed)
        inputs = {'image': torch.rand(size=(3, 512, 512), generator=self.generator, dtype=torch.float32)}
        mask = torch.randint(size=(self.sample_res, self.sample_res), low=0, high=self.num_classes, generator=self.generator, dtype=torch.int64)
        mask = torchvision.transforms.Resize(
            size=(512, 512), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
        )(mask.unsqueeze(0)).squeeze(0)
        labels = {'mask': mask}
        meta_info = {'seed': seed}
        return inputs, labels, meta_info
