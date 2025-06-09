import torch
from data.datasets import COCOStuff164KDataset
from criteria.vision_2d import SemanticSegmentationCriterion


class_dist = torch.Tensor(COCOStuff164KDataset.CLASS_DIST['train']).to(torch.float32)
num_classes = COCOStuff164KDataset.NUM_CLASSES
class_weights = num_classes * (1/class_dist) / torch.sum(1/class_dist)

data_cfg = {
    'train_dataset': {
        'class': COCOStuff164KDataset,
        'args': {
            'data_root': './data/datasets/soft_links/COCOStuff164K',
            'split': 'train2017',
            'semantic_granularity': 'coarse',
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 64,
            'num_workers': 4,
        },
    },
    'criterion': {
        'class': SemanticSegmentationCriterion,
        'args': {
            'class_weights': tuple(class_weights.tolist()),
        },
    },
}
