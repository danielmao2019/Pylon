"""Unified backend for the dataset viewer.

This module combines dataset management, transform management, and state management
into a single simplified backend.
"""
from typing import Dict, Any, List, Optional, Literal
import os
import logging
import importlib.util
from data.transforms.compose import Compose
from utils.builders import build_from_config


# Dataset type definitions
DatasetType = Literal['semseg', '2dcd', '3dcd', 'pcr']

# Dataset groupings
DATASET_GROUPS = {
    'semseg': ['coco_stuff_164k'],
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3dcd': ['urb3dcd', 'slpccd'],
    'pcr': ['synth_pcr', 'real_pcr', 'kitti'],
}

# Dataset format specifications by type
DATASET_FORMATS = {
    'semseg': {
        'input_format': {'image': ['image']},
        'label_format': ['label']
    },
    '2dcd': {
        'input_format': {'image': ['img_1', 'img_2']},
        'label_format': ['change_map']
    },
    '3dcd': {
        'input_format': {'point_cloud': ['pc_1', 'pc_2']},
        'label_format': ['change_map']
    },
    'pcr': {
        'input_format': {
            'point_cloud': ['src_pc', 'tgt_pc'],
            'optional': ['correspondences']
        },
        'label_format': ['transform']
    },
}


class ViewerBackend:
    """Unified backend for the dataset viewer."""

    def __init__(self):
        """Initialize the viewer backend."""
        self.logger = logging.getLogger(__name__)

        # Dataset storage
        self._datasets: Dict[str, Any] = {}
        self._configs: Dict[str, Any] = {}

        # Transform storage (normalized format)
        self._transforms: List[Dict[str, Any]] = []

        # State management
        self.current_dataset: Optional[str] = None
        self.current_index: int = 0
        self.point_size: float = 2.0
        self.point_opacity: float = 0.8
        self.radius: float = 0.05
        self.correspondence_radius: float = 0.1

        # Initialize dataset configurations
        self._init_dataset_configs()

    def _init_dataset_configs(self) -> None:
        """Initialize dataset configurations from config directories."""
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
        config_dirs = {
            'semseg': os.path.join(repo_root, 'configs/common/datasets/semantic_segmentation/train'),
            '2dcd': os.path.join(repo_root, 'configs/common/datasets/change_detection/train'),
            '3dcd': os.path.join(repo_root, 'configs/common/datasets/change_detection/train'),
            'pcr': os.path.join(repo_root, 'configs/common/datasets/point_cloud_registration/train'),
        }

        # Load dataset configurations
        for dataset_type, config_dir in config_dirs.items():
            if not os.path.exists(config_dir):
                self.logger.warning(f"Config directory not found: {config_dir}")
                continue

            for dataset in DATASET_GROUPS.get(dataset_type, []):
                config_file = os.path.join(config_dir, f'{dataset}_data_cfg.py')
                if os.path.exists(config_file):
                    config_name = f"{dataset_type}/{dataset}"
                    self._configs[config_name] = {
                        'path': config_file,
                        'type': dataset_type,
                        'name': dataset
                    }

    def get_available_datasets(self) -> Dict[str, str]:
        """Get available datasets grouped by type.

        Returns:
            Dictionary mapping dataset names to display names
        """
        available = {}
        for config_name in sorted(self._configs.keys()):
            dataset_type, dataset_name = config_name.split('/')
            display_name = f"[{dataset_type.upper()}] {dataset_name}"
            available[config_name] = display_name
        return available

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset and return its information.

        Args:
            dataset_name: Name of the dataset to load (format: "type/name")

        Returns:
            Dataset info dictionary
        """
        # Load dataset if not already loaded
        if dataset_name not in self._datasets:
            config_info = self._configs.get(dataset_name)
            if not config_info:
                raise ValueError(f"Dataset not found: {dataset_name}")

            # Load the config module
            spec = importlib.util.spec_from_file_location("config", config_info['path'])
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # Get the dataset config following the same pattern as main.py
            data_cfg = config_module.data_cfg
            dataset_cfg = data_cfg['train_dataset']

            # Build the dataset
            dataset = build_from_config(dataset_cfg)
            self._datasets[dataset_name] = dataset

        dataset = self._datasets[dataset_name]
        dataset_type = self.get_dataset_type(dataset_name)

        # Extract transforms from dataset config
        transforms_cfg = getattr(dataset.transforms, '__dict__', {})
        if hasattr(dataset.transforms, 'transforms'):
            # Register transforms
            self._clear_transforms()
            for transform in dataset.transforms.transforms:
                self._register_transform(transform)

        # Get dataset info
        self.current_dataset = dataset_name
        return {
            'name': dataset_name,
            'type': dataset_type,
            'length': len(dataset),
            'class_labels': getattr(dataset, 'class_labels', {}),
            'transforms': self.get_available_transforms(),
            'input_format': DATASET_FORMATS[dataset_type]['input_format'],
            'label_format': DATASET_FORMATS[dataset_type]['label_format']
        }

    def get_dataset_type(self, dataset_name: str) -> DatasetType:
        """Determine the type of dataset based on its name."""
        if '/' in dataset_name:
            dataset_type = dataset_name.split('/')[0]
            if dataset_type in DATASET_GROUPS:
                return dataset_type

        # Check by base name
        base_name = dataset_name.split('/')[-1]
        for dataset_type, datasets in DATASET_GROUPS.items():
            if base_name in datasets:
                return dataset_type

        raise ValueError(f"Unknown dataset type for dataset: {dataset_name}")

    def get_datapoint(self, dataset_name: str, index: int, transform_indices: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
        """Get a datapoint with optional transforms applied.

        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            transform_indices: Optional list of transform indices to apply

        Returns:
            Datapoint dictionary with 'inputs', 'labels', and 'meta_info'
        """
        if dataset_name not in self._datasets:
            raise ValueError(f"Dataset not loaded: {dataset_name}")

        dataset = self._datasets[dataset_name]

        # Get raw datapoint
        inputs, labels, meta_info = dataset._load_datapoint(index)
        datapoint = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info
        }

        # Apply transforms if specified
        if transform_indices:
            datapoint = self._apply_transforms(datapoint, transform_indices)

        return datapoint

    def _clear_transforms(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()

    def _register_transform(self, transform: Dict[str, Any]) -> None:
        """Register a normalized transform.

        Args:
            transform: Normalized transform dictionary with 'op', 'input_names', 'output_names'
        """
        self._transforms.append(transform)

    def get_available_transforms(self) -> List[Dict[str, Any]]:
        """Get information about all available transforms."""
        transforms = []
        for i, transform in enumerate(self._transforms):
            transform_op = transform['op']

            # Extract transform name and string representation
            if callable(transform_op):
                transform_name = transform_op.__class__.__name__
                try:
                    transform_string = str(transform_op)
                except Exception:
                    transform_string = transform_name
            elif isinstance(transform_op, dict) and 'class' in transform_op:
                transform_class = transform_op['class']
                transform_name = transform_class.__name__ if hasattr(transform_class, '__name__') else str(transform_class)

                # Format the dict config as a string
                from data.transforms.base_transform import BaseTransform
                args = transform_op.get('args', {})
                formatted_params = BaseTransform.format_params(args)

                if formatted_params:
                    transform_string = f"{transform_name}({formatted_params})"
                else:
                    transform_string = transform_name
            else:
                # This should never happen with normalized transforms
                assert False, f"Should not reach here. Unexpected transform_op type: {type(transform_op)}"

            transforms.append({
                'index': i,
                'name': transform_name,
                'string': transform_string,
                'input_names': transform['input_names'],
                'output_names': transform['output_names']
            })

        return transforms

    def _apply_transforms(self, datapoint: Dict[str, Dict[str, Any]], transform_indices: List[int]) -> Dict[str, Dict[str, Any]]:
        """Apply selected transforms to a datapoint.

        Args:
            datapoint: The datapoint to transform
            transform_indices: List of transform indices to apply

        Returns:
            Transformed datapoint
        """
        # Select the transforms in the order specified by indices
        selected_transforms = [self._transforms[idx] for idx in transform_indices]
        compose = Compose(transforms=[])  # Create empty compose
        compose.transforms = selected_transforms  # Directly assign normalized transforms
        return compose(datapoint)

    def update_state(self, **kwargs) -> None:
        """Update backend state with provided values.

        Args:
            **kwargs: State values to update (current_index, point_size, point_opacity, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_state(self) -> Dict[str, Any]:
        """Get current backend state.

        Returns:
            Dictionary containing current state
        """
        return {
            'current_dataset': self.current_dataset,
            'current_index': self.current_index,
            'point_size': self.point_size,
            'point_opacity': self.point_opacity,
            'radius': self.radius,
            'correspondence_radius': self.correspondence_radius
        }