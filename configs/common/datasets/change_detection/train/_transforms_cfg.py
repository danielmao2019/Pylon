from typing import Tuple, Optional
import torchvision
import data


def transforms_cfg(
    first: Optional[str] = "RandomCrop",
    size: Optional[Tuple[int, int]] = (224, 224),
    resize: Optional[Tuple[int, int]] = None,
) -> dict:
    if first == "RandomCrop":
        first_transform = (
            {
                'class': data.transforms.vision_2d.RandomCrop,
                'args': {'size': size, 'resize': resize, 'interpolation': None},
            },
            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
        )
    elif first == "ResizeMaps":
        first_transform = (
            {
                'class': data.transforms.vision_2d.ResizeMaps,
                'args': {'size': size, 'interpolation': None, 'antialias': True},
            },
            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
        )
    else:
        raise NotImplementedError
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                first_transform,
                (
                    {
                        'class': data.transforms.vision_2d.RandomRotation,
                        'args': {'choices': [0, 90, 180, 270]},
                    },
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    {
                        'class': data.transforms.Randomize,
                        'args': {'transform': data.transforms.vision_2d.Flip(axis=-1), 'p': 0.5},
                    },
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    {
                        'class': data.transforms.Randomize,
                        'args': {'transform': data.transforms.vision_2d.Flip(axis=-2), 'p': 0.5},
                    },
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    {
                        'class': data.transforms.Randomize,
                        'args': {
                            'transform': torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                            'p': 0.5,
                        },
                    },
                    ('inputs', 'img_1'),
                ),
                (
                    {
                        'class': data.transforms.Randomize,
                        'args': {
                            'transform': torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                            'p': 0.5,
                        },
                    },
                    ('inputs', 'img_2'),
                ),
            ],
        },
    }
