import torch
import torchvision
import datasets
import criteria
import metrics
import grad_measures


transforms_config = {
    'class': datasets.transforms.Compose,
    'args': {
        'transforms': [
            (
                torchvision.transforms.Resize(size=(512,)*2, antialias=True),
                ('inputs', 'image'),
            ),
            (
                datasets.transforms.resize.ResizeMaps(size=(512,)*2, antialias=True),
                ('labels', 'depth_estimation'),
            ),
            (
                datasets.transforms.resize.ResizeNormals(target_size=(512,)*2),
                ('labels', 'normal_estimation'),
            ),
            (
                datasets.transforms.resize.ResizeMaps(size=(512,)*2, interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST),
                ('labels', 'semantic_segmentation'),
            ),
        ],
    },
}

collate_fn_config = {
    'class': datasets.collators.BaseCollator,
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
        'class': datasets.NYUv2Dataset,
        'args': {
            'data_root': "./datasets/datasets/soft_links/NYUD_MT",
            'split': "train",
            'indices': None,
            'transforms': transforms_config,
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
        'class': datasets.NYUv2Dataset,
        'args': {
            'data_root': "./datasets/datasets/soft_links/NYUD_MT",
            'split': "val",
            'indices': None,
            'transforms': transforms_config,
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
        'class': datasets.NYUv2Dataset,
        'args': {
            'data_root': "./datasets/datasets/soft_links/NYUD_MT",
            'split': "test",
            'indices': None,
            'transforms': transforms_config,
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
        'class': criteria.MultiTaskCriterion,
        'args': {
            'criterion_configs': {
                'depth_estimation': {
                    'class': criteria.DepthEstimationCriterion,
                    'args': {},
                },
                'normal_estimation': {
                    'class': criteria.NormalEstimationCriterion,
                    'args': {},
                },
                'semantic_segmentation': {
                    'class': criteria.SemanticSegmentationCriterion,
                    'args': {
                        'ignore_index': datasets.NYUv2Dataset.IGNORE_INDEX,
                    },
                },
            },
        },
    },
    'metric': {
        'class': metrics.XMTLMetric,
        'args': {
            'metric_configs': {
                'depth_estimation': {
                    'class': metrics.DepthEstimationMetric,
                    'args': {},
                },
                'normal_estimation': {
                    'class': metrics.NormalEstimationMetric,
                    'args': {},
                },
                'semantic_segmentation': {
                    'class': metrics.SemanticSegmentationMetric,
                    'args': {
                        'num_classes': datasets.NYUv2Dataset.NUM_CLASSES,
                        'ignore_index': datasets.NYUv2Dataset.IGNORE_INDEX,
                    },
                },
            },
            'grad_measures': {
                'grad_conflicts': grad_measures.grad_conflicts,
                'grad_dominance': grad_measures.grad_dominance,
                'grad_stability': grad_measures.grad_stability,
                'feature_ent': grad_measures.feature_ent,
                'task_ent_cos_sim': grad_measures.task_ent_cos_sim,
                'task_ent_sym_cross_entropy': grad_measures.task_ent_sym_cross_entropy,
                'task_ent_sym_Kullback_Leibler_divergence': grad_measures.task_ent_sym_Kullback_Leibler_divergence,
                'task_ent_Jensen_Shannon_divergence': grad_measures.task_ent_Jensen_Shannon_divergence,
                'task_ent_earth_mover_distance': grad_measures.task_ent_earth_mover_distance,
            },
        },
    },
}
