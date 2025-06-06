from typing import List, Callable, Union
import random
import torch
from data.transforms import BaseTransform, Compose


class Randomize(BaseTransform):

    def __init__(self, transform: Callable, p: float) -> None:
        assert callable(transform), f"{type(callable)=}"
        assert type(transform) not in {BaseTransform, Compose}, f"{type(transform)=}"
        self.transform = transform
        assert type(p) in [int, float], f"{type(p)=}"
        assert 0 <= p <= 1, f"{p=}"
        self.p = p
        self.generator = random.Random()

    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.generator.uniform(0, 1) < self.p:
            return self.transform(*args)
        else:
            if len(args) == 1:
                return args[0]
            else:
                return list(args)
