from typing import List, Union
import torch
from data.transforms.base_transform import BaseTransform


class Identity(BaseTransform):

    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        return args
