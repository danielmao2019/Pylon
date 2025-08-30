import easydict as edict
import data


train_dataloader_cfg = {
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
        'shuffle': True,
    },
}

val_dataloader_cfg = {
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
        'shuffle': False,
    },
}
