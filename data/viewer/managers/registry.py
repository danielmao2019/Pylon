"""Dataset registry module.

This module contains centralized definitions for dataset groups and formats
used across the codebase.
"""
from typing import Dict, List, Literal

# Dataset type definitions
DatasetType = Literal['2d_change_detection', '3d_change_detection', 'point_cloud_registration']

# Dataset groupings
DATASET_GROUPS = {
    '2d_change_detection': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3d_change_detection': ['urb3dcd', 'slpccd'],
    'point_cloud_registration': ['synth_pcr', 'real_pcr', 'kitti'],
}

# Dataset format specifications by type
DATASET_FORMATS = {
    '2d_change_detection': {
        'input_format': {
            'image': ['img_1', 'img_2']
        },
        'label_format': ['change_map']
    },
    '3d_change_detection': {
        'input_format': {
            'point_cloud': ['pc_1', 'pc_2']
        },
        'label_format': ['change_map']
    },
    'point_cloud_registration': {
        'input_format': {
            'point_cloud': ['src_pc', 'tgt_pc'],
            'optional': ['correspondences']
        },
        'label_format': ['transform']
    }
}

def get_all_supported_datasets() -> Dict[str, List[str]]:
    """Get a mapping of all supported datasets by their type.
    
    Returns:
        Dictionary mapping dataset types to lists of supported dataset names
    """
    return DATASET_GROUPS

def get_dataset_type(dataset_name: str) -> DatasetType:
    """Determine the type of dataset based on its name.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dataset type (2d_change_detection, 3d_change_detection, or point_cloud_registration)

    Raises:
        ValueError: If the dataset type cannot be determined
    """
    base_name = dataset_name.split('/')[-1]

    for dataset_type, datasets in DATASET_GROUPS.items():
        if base_name in datasets:
            return dataset_type

    # Raise error for unknown dataset types
    raise ValueError(f"Unknown dataset type for dataset: {dataset_name}")
