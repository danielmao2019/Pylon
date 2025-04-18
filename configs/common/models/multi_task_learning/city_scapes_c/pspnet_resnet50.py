import data
import models


model_config_depth_estimation = {
    'class': models.multi_task_learning.CityScapes_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['depth_estimation']),
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_semantic_segmentation = {
    'class': models.multi_task_learning.CityScapes_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['semantic_segmentation']),
        'num_classes': data.datasets.CityScapesDataset.NUM_CLASSES_C,
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_instance_segmentation = {
    'class': models.multi_task_learning.CityScapes_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(['instance_segmentation']),
        'return_shared_rep': False,
        'use_attention': False,
    },
}

model_config_all_tasks = {
    'class': models.multi_task_learning.CityScapes_PSPNet,
    'args': {
        'backbone': models.backbones.resnet50(weights='DEFAULT'),
        'in_channels': 2048,
        'tasks': set(["depth_estimation", "semantic_segmentation", "instance_segmentation"]),
        'num_classes': data.datasets.CityScapesDataset.NUM_CLASSES_C,
        'return_shared_rep': True,
        'use_attention': False,
    },
}
