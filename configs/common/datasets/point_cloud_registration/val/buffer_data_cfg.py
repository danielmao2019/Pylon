import easydict as edict
import data


data_cfg = {
    'val_dataset': {
        'class': data.datasets.KITTIDataset,
        'args': {
            'data_root': './data/datasets/soft_links/KITTI',
            'split': 'val',
            'transforms_cfg': None,
        },
    },
    'val_dataloader': {
        'class': data.dataloaders.BufferDataloader,
        'args': {
            'config': edict.EasyDict({
                'point': {
                    'conv_radius': 2.0,
                },
                'data': {
                    'voxel_size_0': 0.30,
                },
            }),
            'batch_size': 1,
        },
    },
    'test_dataset': None,
    'test_dataloader': None,
}
