import torch
import torchvision
import data
import criteria
import metrics


transforms_cfg = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                {
                    'class': torchvision.transforms.Resize,
                    'args': {'size': (288, 384), 'antialias': True},
                },
                ('inputs', 'image'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (288, 384), 'antialias': True},
                },
                ('labels', 'depth_estimation'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeNormals,
                    'args': {'target_size': (288, 384)},
                },
                ('labels', 'normal_estimation'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (288, 384), 'interpolation': 'nearest', 'antialias': True},
                },
                ('labels', 'semantic_segmentation'),
            ),
        ],
    },
}

collate_fn_cfg = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_resolution': torch.tensor,
            },
        },
    },
}

config = {
    'train_dataset': {
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "train",
            'indices': None,
            'transforms_cfg': transforms_cfg,
            'semantic_granularity': 'coarse',
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 32,
            'num_workers': 8,
            'collate_fn': collate_fn_cfg
        },
    },
    'val_dataset': {
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_cfg,
            'semantic_granularity': 'coarse',
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
            'collate_fn': collate_fn_cfg,
        },
    },
    'test_dataset': {
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_cfg,
            'semantic_granularity': 'coarse',
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
            'collate_fn': collate_fn_cfg,
        },
    },
    'criterion': {
        'class': criteria.wrappers.MultiTaskCriterion,
        'args': {
            'criterion_configs': {
                'depth_estimation': {
                    'class': criteria.vision_2d.DepthEstimationCriterion,
                    'args': {},
                },
                'normal_estimation': {
                    'class': criteria.vision_2d.NormalEstimationCriterion,
                    'args': {},
                },
                'semantic_segmentation': {
                    'class': criteria.vision_2d.SemanticSegmentationCriterion,
                    'args': {
                        'ignore_index': data.datasets.NYUv2Dataset.IGNORE_INDEX,
                    },
                },
            },
        },
    },
    'metric': {
        'class': metrics.wrappers.MultiTaskMetric,
        'args': {
            'metric_configs': {
                'depth_estimation': {
                    'class': metrics.vision_2d.DepthEstimationMetric,
                    'args': {},
                },
                'normal_estimation': {
                    'class': metrics.vision_2d.NormalEstimationMetric,
                    'args': {},
                },
                'semantic_segmentation': {
                    'class': metrics.vision_2d.SemanticSegmentationMetric,
                    'args': {
                        'num_classes': data.datasets.NYUv2Dataset.NUM_CLASSES_C,
                        'ignore_index': data.datasets.NYUv2Dataset.IGNORE_INDEX,
                    },
                },
            },
        },
    },
}
