from typing import List, Union, Callable, Optional, Any
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

    def __call__(self, *args, seed: Optional[Any] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        generator = self._get_generator(g_type='random', seed=seed)
        if generator.uniform(0, 1) < self.p:
            try:
                return self.transform(*args, seed=seed)
            except Exception as e:
                # If error is about unexpected seed argument, try without seed
                if "got an unexpected keyword argument 'seed'" in str(e):
                    if len(args) == 1:
                        return self.transform(args[0])
                    else:
                        return self.transform(*args)
                else:
                    raise
        else:
            if len(args) == 1:
                return args[0]
            else:
                return list(args)
