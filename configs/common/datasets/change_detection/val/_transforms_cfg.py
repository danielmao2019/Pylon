from typing import Tuple
import data


def transforms_cfg(size: Tuple[int, int]) -> dict:
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                (
                    data.transforms.crop.RandomCrop(size=size),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
                ),
            ],
        },
    }
