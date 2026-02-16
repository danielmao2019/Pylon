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
from data.viewer.utils.settings_config import ViewerSettings


# Dataset type definition for backward compatibility with eval viewer
DatasetType = Literal['semseg', '2dcd', '3dcd', 'pcr', 'mtl', 'ivision', 'general']

# Dataset groupings by type for UI organization
DATASET_GROUPS = {
    'semseg': ['coco_stuff_164k', 'whu_bd'],
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd', 'ivision_2dcd_original', 'ivision_2dcd_synthetic'],
    '3dcd': ['urb3dcd', 'slpccd', 'ivision_3dcd'],
    'pcr': [
        'kitti', 'threedmatch', 'threedlomatch', 'modelnet40', 'lidar_camera_pose_pcr',
        'buffer',
        'ivision_pcr', 'geotransformer_ivision_pcr', 'overlappredator_ivision_pcr',
    ],
    'mtl': [
        'multi_mnist', 'celeb_a', 'multi_task_facial_landmark',
        'nyu_v2_c', 'nyu_v2_f', 'city_scapes_c', 'city_scapes_f', 'pascal_context',
        'ade_20k',
    ],
    'ivision': ['ivision_image'],
    'general': ['BaseRandomDataset'],  # General-purpose datasets for testing
}

# Registry of dataset classes that require 3D visualization
REQUIRES_3D_CLASSES = [
    'Base3DCDDataset',
    'BasePCRDataset',
    'Buffer3DDataset',
    'KITTIDataset',
    'ThreeDMatchDataset',
    'ThreeDLoMatchDataset',
    'ModelNet40Dataset',
    'BufferDataset',
    'URB3DCDDataset',
    'SLPCCDDataset',
    'iVISION_PCR_DATASET',
    'iVISION_3DCD_Dataset',
    'iVISION_2DCD_Dataset',
]



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

        # Initialize 3D settings from centralized configuration
        default_settings = ViewerSettings.DEFAULT_3D_SETTINGS
        self.point_size: float = default_settings['point_size']
        self.point_opacity: float = default_settings['point_opacity']
        self.sym_diff_radius: float = default_settings['sym_diff_radius']
        self.lod_type: str = default_settings['lod_type']

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
            'mtl': os.path.join(repo_root, 'configs/common/datasets/multi_task_learning/train'),
            'ivision': os.path.join(repo_root, 'configs/common/datasets/ivision'),
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

    def get_available_datasets_hierarchical(self) -> Dict[str, Dict[str, str]]:
        """Get available datasets grouped hierarchically by type.

        Returns:
            Dictionary mapping dataset types to their datasets
        """
        hierarchical = {}
        for config_name in sorted(self._configs.keys()):
            dataset_type, dataset_name = config_name.split('/')
            if dataset_type not in hierarchical:
                hierarchical[dataset_type] = {}
            hierarchical[dataset_type][config_name] = dataset_name
        return hierarchical

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
            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load config from path: {config_info['path']}")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            # Try to load dataset config - support both old and new formats
            dataset_cfg = None

            # Try new format first (train_dataset_cfg directly in module)
            if hasattr(config_module, 'train_dataset_cfg'):
                dataset_cfg = config_module.train_dataset_cfg
            # Fall back to old format (data_cfg['train_dataset'])
            elif hasattr(config_module, 'data_cfg'):
                self.logger.warning(f"DEPRECATED: {config_info['path']} uses old config format. Please migrate to new format with train_dataset_cfg, val_dataset_cfg, test_dataset_cfg")
                data_cfg = config_module.data_cfg
                dataset_cfg = data_cfg['train_dataset']
            else:
                raise ValueError(f"Config file {config_info['path']} has neither 'train_dataset_cfg' (new format) nor 'data_cfg' (old format)")

            if dataset_cfg is None:
                raise ValueError(f"Could not load dataset configuration from {config_info['path']}")

            # Build the dataset
            dataset = build_from_config(dataset_cfg)
            self._datasets[dataset_name] = dataset

        dataset = self._datasets[dataset_name]

        # Extract and register transforms for viewer, then clear them from dataset
        if hasattr(dataset.transforms, 'transforms'):
            # Register transforms for viewer control
            self._clear_transforms()
            for transform in dataset.transforms.transforms:
                self._register_transform(transform)

            # Clear transforms from dataset - viewer will handle all transform application
            dataset.transforms = Compose(transforms=[])

        # Get dataset info
        self.current_dataset = dataset_name
        return {
            'name': dataset_name,
            'type': self._get_dataset_type(dataset_name),
            'length': len(dataset),
            'class_labels': getattr(dataset, 'class_labels', {}),
            'transforms': self.get_available_transforms(),
            'requires_3d_visualization': self._requires_3d_visualization(dataset)
        }

    def _get_dataset_type(self, dataset_name: str) -> str:
        """Determine the type of dataset for UI organization.

        Args:
            dataset_name: Name of the dataset to analyze

        Returns:
            Dataset type string for UI grouping
        """
        # Ensure dataset is loaded
        if dataset_name not in self._datasets:
            raise ValueError(f"Dataset not loaded: {dataset_name}")

        dataset = self._datasets[dataset_name]
        return self._get_dataset_type_from_inheritance(dataset)

    def get_datapoint(self, dataset_name: str, index: int, transform_indices: List[int]) -> Dict[str, Dict[str, Any]]:
        """Get a datapoint with transforms applied.

        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            transform_indices: List of transform indices to apply (empty list means no transforms)

        Returns:
            Datapoint dictionary with 'inputs', 'labels', and 'meta_info'
        """
        # Input validation
        assert isinstance(dataset_name, str), f"dataset_name must be str, got {type(dataset_name)}"
        assert isinstance(index, int), f"index must be int, got {type(index)}"
        assert isinstance(transform_indices, list), f"transform_indices must be list, got {type(transform_indices)}"
        assert all(isinstance(idx, int) for idx in transform_indices), f"All transform indices must be int, got {transform_indices}"

        if dataset_name not in self._datasets:
            raise ValueError(f"Dataset not loaded: {dataset_name}")

        dataset = self._datasets[dataset_name]

        # Get datapoint with device transfer (no transforms applied - cleared during initialization)
        datapoint = dataset[index]

        # Apply transforms (always call _apply_transforms, it handles empty list correctly)
        datapoint = self._apply_transforms(datapoint, transform_indices, index)

        return datapoint

    def get_dataset_instance(self, dataset_name: str) -> Any:
        """Get the dataset instance for custom display methods.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset is not loaded
        """
        assert isinstance(dataset_name, str), f"dataset_name must be str, got {type(dataset_name)}"

        if dataset_name not in self._datasets:
            raise ValueError(f"Dataset not loaded: {dataset_name}")

        return self._datasets[dataset_name]

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

    def _apply_transforms(self, datapoint: Dict[str, Dict[str, Any]], transform_indices: List[int], datapoint_index: int) -> Dict[str, Dict[str, Any]]:
        """Apply selected transforms to a datapoint with deterministic seeding.

        Args:
            datapoint: The datapoint to transform
            transform_indices: List of transform indices to apply
            datapoint_index: Index of the datapoint for deterministic seeding

        Returns:
            Transformed datapoint
        """
        # Handle empty transform list case
        if not transform_indices:
            return datapoint

        # Select the transforms in the order specified by indices
        # Convert stored transform dicts to the format expected by Compose
        selected_transforms = []
        for idx in transform_indices:
            transform_dict = self._transforms[idx]
            # Compose expects the dictionary format with 'op', 'input_names', 'output_names'
            selected_transforms.append(transform_dict)

        # Use Compose with type annotation to suppress the warning
        compose = Compose(transforms=selected_transforms)
        # Apply transforms with deterministic seed using datapoint index
        return compose(datapoint, seed=(0, datapoint_index))

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
            'sym_diff_radius': self.sym_diff_radius,
            'lod_type': self.lod_type
        }

    def _requires_3d_visualization(self, dataset: Any) -> bool:
        """Check if a dataset requires 3D visualization using hardcoded class registry.

        Args:
            dataset: Dataset instance

        Returns:
            True if dataset requires 3D visualization
        """
        dataset_class_name = type(dataset).__name__
        return dataset_class_name in REQUIRES_3D_CLASSES

    def _get_dataset_type_from_inheritance(self, dataset: Any) -> str:
        """Get dataset type based on class inheritance for fallback cases.

        Args:
            dataset: Dataset instance

        Returns:
            Dataset type string
        """
        # Import the base display classes to check inheritance
        from data.datasets.change_detection_datasets.base_2dcd_dataset import Base2DCDDataset
        from data.datasets.change_detection_datasets.base_3dcd_dataset import Base3DCDDataset
        from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
        from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import BaseSemsegDataset
        from data.datasets.multi_task_datasets.base_multi_task_dataset import BaseMultiTaskDataset

        # Check inheritance hierarchy to determine type
        if isinstance(dataset, BaseMultiTaskDataset):
            return 'mtl'
        elif isinstance(dataset, Base2DCDDataset):
            return '2dcd'
        elif isinstance(dataset, Base3DCDDataset):
            return '3dcd'
        elif isinstance(dataset, BasePCRDataset):
            return 'pcr'
        elif isinstance(dataset, BaseSemsegDataset):
            return 'semseg'
        else:
            return 'general'
