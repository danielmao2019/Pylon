from typing import Optional
import math
import torch
from .overfit_base_dataset import OverfitBaseDataset


class SemanticSegmentationOverfitDataset(OverfitBaseDataset):

    def __init__(
        self,
        num_classes: int,
        num_examples: int,
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
                    {'size': (3, 512, 512), 'dtype': torch.float32},
                ),
            },
            'labels': {
                'mask': (
                    torch.randint,
                    {'size': (math.ceil(math.sqrt(num_classes)),) * 2, 'low': 0, 'high': num_classes, 'dtype': torch.int64}
                ),
            },
        }
        super(SemanticSegmentationOverfitDataset, self).__init__(
            num_examples=num_examples, gen_func_config=gen_func_config, initial_seed=initial_seed,
        )
