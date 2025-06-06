from typing import Tuple, Optional
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
                'args': {'size': size, 'interpolation': None},
            },
            [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
        )
    else:
        raise NotImplementedError
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [first_transform],
        },
    }
