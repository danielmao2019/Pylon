import torch
import data
import criteria


transforms_cfg = {
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

collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_size': torch.Tensor,
                'crop_loc': torch.Tensor,
                'crop_size': torch.Tensor,
            },
        },
    },
}

class_dist = torch.Tensor(data.datasets.AirChangeDataset.CLASS_DIST['train'], dtype=torch.float32)
num_classes = data.datasets.AirChangeDataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

config = {
    'train_dataset': {
        'class': data.datasets.AirChangeDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/AirChange",
            'split': "train",
            'transforms_cfg': transforms_cfg,
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 4,
            'num_workers': 4,
            'collate_fn': collate_fn_cfg,
        },
    },
    'criterion': {
        'class': criteria.vision_2d.SemanticSegmentationCriterion,
        'args': {
            'class_weights': tuple(class_weights.tolist()),
        },
    },
}
