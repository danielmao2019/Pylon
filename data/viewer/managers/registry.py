"""Dataset registry module.

This module contains centralized definitions for dataset groups and formats
used across the codebase.
"""
from typing import Literal
import os


# Dataset type definitions
DatasetType = Literal['2dcd', '3dcd', 'pcr']

# Dataset groupings
DATASET_GROUPS = {
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3dcd': ['urb3dcd', 'slpccd'],
    'pcr': ['synth_pcr', 'real_pcr', 'kitti'],
}

repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG_DIRS = {
    '2dcd': os.path.join(repo_root, 'configs/common/datasets/change_detection/train'),
    '3dcd': os.path.join(repo_root, 'configs/common/datasets/change_detection/train'),
    'pcr': os.path.join(repo_root, 'configs/common/datasets/point_cloud_registration/train'),
}

# Dataset format specifications by type
DATASET_FORMATS = {
    '2dcd': {
        'input_format': {
            'image': ['img_1', 'img_2']
        },
        'label_format': ['change_map']
    },
    '3dcd': {
        'input_format': {
            'point_cloud': ['pc_1', 'pc_2']
        },
        'label_format': ['change_map']
    },
    'pcr': {
        'input_format': {
            'point_cloud': ['src_pc', 'tgt_pc'],
            'optional': ['correspondences']
        },
        'label_format': ['transform']
    }
}


def get_dataset_type(dataset_name: str) -> DatasetType:
    """Determine the type of dataset based on its name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset type (2dcd, 3dcd, pcr, etc.)

    Raises:
        ValueError: If the dataset type cannot be determined
    """
    base_name = dataset_name.split('/')[-1]

    for dataset_type, datasets in DATASET_GROUPS.items():
        if base_name in datasets:
            return dataset_type

    # Raise error for unknown dataset types
    raise ValueError(f"Unknown dataset type for dataset: {dataset_name}. {base_name=}, {DATASET_GROUPS=}.")
