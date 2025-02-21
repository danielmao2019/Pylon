from typing import Tuple, Optional
import data


def transforms_cfg(
    first: Optional[str] = "RandomCrop",
    size: Optional[Tuple[int, int]] = (224, 224),
    resize: Optional[Tuple[int, int]] = None,
) -> dict:
    if first == "RandomCrop":
        first_transform = [(
            data.transforms.crop.RandomCrop(size=size, resize=resize, interpolation='bilinear'),
            [('inputs', 'img_1'), ('inputs', 'img_2')],
        ), (
            data.transforms.crop.RandomCrop(size=size, resize=resize, interpolation='nearest'),
            [('labels', 'change_map')],
        )]
    elif first == "ResizeMaps":
        first_transform = [(
            data.transforms.resize.ResizeMaps(size=size, interpolation='bilinear'),
            [('inputs', 'img_1'), ('inputs', 'img_2')],
        ), (
            data.transforms.resize.ResizeMaps(size=size, interpolation='nearest'),
            [('labels', 'change_map')],
        )]
    else:
        raise NotImplementedError
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': first_transform,
        },
    }
