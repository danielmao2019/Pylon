import torch
import data
import criteria


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                data.transforms.resize.ResizeMaps(size=(256, 256), antialias=True),
                [('inputs', 'img_1'), ('inputs', 'img_2'), ('labels', 'change_map')]
            ),
        ],
    },
}

collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {},
    },
}

source_dataset = data.datasets.Bi2SingleTemporal(
    source=data.datasets.SYSU_CD_Dataset(data_root="./data/datasets/soft_links/SYSU-CD", split="train"),
)

config = {
    'train_dataset': {
        'class': data.datasets.I3PEDataset,
        'args': {
            'source': source_dataset,
            'dataset_size': len(source_dataset),
            'exchange_ratio': 0.75,
            'transforms_cfg': transforms_config,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 128,
            'num_workers': 8,
            'collate_fn': collate_fn_config,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SymmetricChangeDetectionCriterion,
        'args': {
            'class_weights': tuple((
                data.datasets.SYSU_CD_Dataset.NUM_CLASSES*(1/torch.tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']))/torch.sum(1/torch.tensor(data.datasets.SYSU_CD_Dataset.CLASS_DIST['train']))
            ).tolist()),
        },
    },
}
