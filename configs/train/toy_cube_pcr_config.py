from torch.utils.data.dataloader import DataLoader
from data.datasets.pcr_datasets.toy_cube_dataset import ToyCubeDataset


config = {
    'dataset': {
        'class': ToyCubeDataset,
        'args': {
            'split': 'train',
            'cube_density': 8,  # 8x8 grid per face = 384 points per cube
        },
    },
    'dataloader': {
        'class': DataLoader,
        'args': {
            'batch_size': 1,
            'num_workers': 1,
            'shuffle': False,
        },
    },
}
