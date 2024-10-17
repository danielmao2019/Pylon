import torch
import torchvision
import data
import criteria
import metrics


transforms_config = {
    'class': data.transforms.Compose,
    'args': {
        'transforms': [
            (
                torchvision.transforms.Resize(size=(256, 512), antialias=True),
                ('inputs', 'image'),
            ),
            (
                data.transforms.resize.ResizeMaps(size=(256, 512), antialias=True),
                ('labels', 'depth_estimation'),
            ),
            (
                data.transforms.resize.ResizeMaps(size=(256, 512), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST),
                ('labels', 'semantic_segmentation'),
            ),
            (
                torchvision.transforms.Resize(size=(256, 512), antialias=True),
                ('labels', 'instance_segmentation'),
            ),
        ],
    },
}

collate_fn_config = {
    'class': data.collators.BaseCollator,
    'args': {
        'collators': {
            'meta_info': {
                'image_resolution': lambda x: torch.tensor(x, dtype=torch.int64),
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
            'transforms_cfg': transforms_config,
            'semantic_granularity': 'fine',
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 32,
            'num_workers': 8,
            'collate_fn': collate_fn_config,
        },
    },
    'val_dataset': {
        'class': data.datasets.CityScapesDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/city-scapes",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_config,
            'semantic_granularity': 'fine',
        },
    },
    'val_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
            'collate_fn': collate_fn_config,
        },
    },
    'test_dataset': {
        'class': data.datasets.CityScapesDataset,
        'args': {
            'data_root': "./data/datasets/soft_links/city-scapes",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_config,
            'semantic_granularity': 'fine',
        },
    },
    'test_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'num_workers': 8,
            'collate_fn': collate_fn_config,
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
                        'num_classes': data.datasets.CityScapesDataset.NUM_CLASSES_F,
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
