from typing import Tuple, Optional
import torch
from .base_random_dataset import BaseRandomDataset


class ClassificationRandomDataset(BaseRandomDataset):

    def __init__(
        self,
        num_classes: int,
        num_examples: int,
        image_res: Tuple[int, int],
        initial_seed: Optional[int] = None,
    ) -> None:
        # init num classes
        assert type(num_classes) == int, f"{type(num_classes)=}"
        assert num_classes > 0, f"{num_classes=}"
        # init gen func config
        gen_func_config = {
            'inputs': {
                'image': (
                    torch.rand,
                    {'size': (3, image_res, image_res), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'target': (
                    torch.randint,
                    {'size': (), 'low': 0, 'high': num_classes, 'dtype': torch.int64}
                ),
            },
        }
        super(ClassificationRandomDataset, self).__init__(
            num_examples=num_examples, gen_func_config=gen_func_config, initial_seed=initial_seed,
        )
