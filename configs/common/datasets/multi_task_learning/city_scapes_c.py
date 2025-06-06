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
                    'args': {'size': (256, 512), 'antialias': True},
                },
                ('inputs', 'image'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (256, 512), 'antialias': True},
                },
                ('labels', 'depth_estimation'),
            ),
            (
                {
                    'class': data.transforms.vision_2d.ResizeMaps,
                    'args': {'size': (256, 512), 'interpolation': 'nearest', 'antialias': True},
                },
                ('labels', 'semantic_segmentation'),
            ),
            (
                {
                    'class': torchvision.transforms.Resize,
                    'args': {'size': (256, 512), 'antialias': True},
                },
                ('labels', 'instance_segmentation'),
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
        'class': data.datasets.CityScapesDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/city-scapes",
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
            'collate_fn': collate_fn_cfg,
        },
    },
    'val_dataset': {
        'class': data.datasets.CityScapesDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/city-scapes",
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
        'class': data.datasets.CityScapesDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/city-scapes",
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
                'semantic_segmentation': {
                    'class': criteria.vision_2d.SemanticSegmentationCriterion,
                    'args': {
                        'ignore_index': data.datasets.CityScapesDataset.IGNORE_INDEX,
                    },
                },
                'instance_segmentation': {
                    'class': criteria.vision_2d.InstanceSegmentationCriterion,
                    'args': {
                        'ignore_index': data.datasets.CityScapesDataset.IGNORE_INDEX,
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
                'semantic_segmentation': {
                    'class': metrics.vision_2d.SemanticSegmentationMetric,
                    'args': {
                        'num_classes': data.datasets.CityScapesDataset.NUM_CLASSES_C,
                        'ignore_index': data.datasets.CityScapesDataset.IGNORE_INDEX,
                    },
                },
                'instance_segmentation': {
                    'class': metrics.vision_2d.InstanceSegmentationMetric,
                    'args': {
                        'ignore_index': data.datasets.CityScapesDataset.IGNORE_INDEX,
                    },
                },
            },
        },
    },
}
