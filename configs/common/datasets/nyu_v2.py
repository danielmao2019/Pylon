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
                torchvision.transforms.Resize(size=(288, 384), antialias=True),
                ('inputs', 'image'),
            ),
            (
                data.transforms.resize.ResizeMaps(size=(288, 384), antialias=True),
                ('labels', 'depth_estimation'),
            ),
            (
                data.transforms.resize.ResizeNormals(target_size=(288, 384)),
                ('labels', 'normal_estimation'),
            ),
            (
                data.transforms.resize.ResizeMaps(size=(288, 384), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST),
                ('labels', 'semantic_segmentation'),
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
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "train",
            'indices': None,
            'transforms_cfg': transforms_config,
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
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "val",
            'indices': None,
            'transforms_cfg': transforms_config,
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
        'class': data.datasets.NYUv2Dataset,
        'args': {
            'data_root': "./data/datasets/soft_links/NYUD_MT",
            'split': "test",
            'indices': None,
            'transforms_cfg': transforms_config,
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
                        'num_classes': data.datasets.NYUv2Dataset.NUM_CLASSES,
                        'ignore_index': data.datasets.NYUv2Dataset.IGNORE_INDEX,
                    },
                },
            },
        },
    },
}
