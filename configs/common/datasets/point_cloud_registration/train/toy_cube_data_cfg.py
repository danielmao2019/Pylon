import data


data_cfg = {
    'train_dataset': {
        'class': data.datasets.ToyCubeDataset,
        'args': {
            'split': 'train',
            'cube_density': 8,  # 8x8 grid per face = 384 points per cube
        },
    },
}
