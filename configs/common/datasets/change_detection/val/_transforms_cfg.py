import data


def transforms_cfg(img_size: int) -> dict:
    return {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                (
                    data.transforms.RandomCrop(size=(img_size, img_size)),
                    [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
                ),
            ],
        },
    }
