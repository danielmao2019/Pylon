import data


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.RandomCrop(size=(224, 224)),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
            ),
        ],
    },
}
