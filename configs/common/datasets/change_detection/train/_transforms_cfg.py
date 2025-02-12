import torchvision
import data


def transforms_cfg(img_size: int) -> dict:
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                (
                    data.transforms.RandomCrop(size=(img_size, img_size)),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    data.transforms.RandomRotation(choices=[0, 90, 180, 270]),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    data.transforms.Randomize(transform=data.transforms.Flip(axis=-1), p=0.5),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    data.transforms.Randomize(transform=data.transforms.Flip(axis=-2), p=0.5),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')],
                ),
                (
                    data.transforms.Randomize(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), p=0.5),
                    ('inputs', 'img_1'),
                ),
                (
                    data.transforms.Randomize(torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), p=0.5),
                    ('inputs', 'img_2'),
                ),
            ],
        },
    }
