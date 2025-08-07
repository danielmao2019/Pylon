"""Tests for backend display functions.

This module tests the critical backend functionality for determining dataset types
and managing dataset instances for the display system using real datasets.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from data.datasets.base_dataset import BaseDataset
from data.datasets.change_detection_datasets.base_2d_cd_dataset import Base2DCDDataset
from data.datasets.change_detection_datasets.base_3d_cd_dataset import Base3DCDDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import (
    LevirCdDataset,
)
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset

# Synthetic dataset imports for 3D change detection (due to device compatibility issues with real datasets)
from data.datasets.random_datasets.base_random_dataset import BaseRandomDataset
from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import (
    BaseSemsegDataset,
)

# Real dataset imports
from data.datasets.semantic_segmentation_datasets.coco_stuff_164k_dataset import (
    COCOStuff164KDataset,
)
from data.viewer.backend.backend import ViewerBackend


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def real_semseg_dataset():
    """Real semantic segmentation dataset for testing."""
    data_root = "./data/datasets/soft_links/COCOStuff164K"

    # Check if dataset is available
    if not os.path.exists(data_root):
        pytest.skip(f"COCOStuff164K dataset not available at {data_root}")

    try:
        dataset = COCOStuff164KDataset(
            data_root=data_root,
            split='val2017',  # Use smaller validation split for testing
            device=torch.device('cpu'),
        )
        return dataset
    except Exception as e:
        pytest.skip(f"Failed to load COCOStuff164K dataset: {e}")


@pytest.fixture
def real_2dcd_dataset():
    """Real 2D change detection dataset for testing."""
    data_root = "./data/datasets/soft_links/LEVIR-CD"

    # Check if dataset is available
    if not os.path.exists(data_root):
        pytest.skip(f"LEVIR-CD dataset not available at {data_root}")

    try:
        dataset = LevirCdDataset(
            data_root=data_root,
            split='test',  # Use test split for testing
            device=torch.device('cpu'),
        )
        return dataset
    except Exception as e:
        pytest.skip(f"Failed to load LEVIR-CD dataset: {e}")


@pytest.fixture
def synthetic_3dcd_dataset():
    """Synthetic 3D change detection dataset for testing.

    Using synthetic data as fallback because real 3D datasets have device compatibility issues in the test environment.
    """

    class _Synthetic3DCDDataset(Base3DCDDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['pc_1', 'pc_2']
        LABEL_NAMES = ['change_map']

        def _init_annotations(self) -> None:
            self.annotations = [
                {'id': 0},
                {'id': 1},
                {'id': 2},
            ]  # Small dataset for testing

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ]:
            n_points = 1000
            # Create more realistic point clouds with spatial structure
            pc1 = {
                'pos': torch.randn(n_points, 3, dtype=torch.float32) * 10
            }  # Scale for realistic coordinates
            pc2 = {'pos': torch.randn(n_points, 3, dtype=torch.float32) * 10}
            change_map = torch.randint(0, 2, (n_points,), dtype=torch.long)
            return {'pc_1': pc1, 'pc_2': pc2}, {'change_map': change_map}, {'idx': idx}

    return _Synthetic3DCDDataset(split='test', device=torch.device('cpu'))


@pytest.fixture
def real_pcr_dataset():
    """Real point cloud registration dataset for testing."""
    data_root = "./data/datasets/soft_links/KITTI"

    # Check if dataset is available
    if not os.path.exists(data_root):
        pytest.skip(f"KITTI dataset not available at {data_root}")

    try:
        dataset = KITTIDataset(
            data_root=data_root,
            split='val',  # Use validation split for testing
            device=torch.device('cpu'),
        )
        return dataset
    except Exception as e:
        pytest.skip(f"Failed to load KITTI dataset: {e}")


@pytest.fixture
def non_display_dataset():
    """Dataset that doesn't inherit from any base display class for error testing."""

    class _NonDisplayDataset(BaseDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['data']
        LABEL_NAMES = ['target']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}, {'id': 1}]  # Small test dataset

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ]:
            data = torch.randn(10, dtype=torch.float32)
            target = torch.tensor(1, dtype=torch.long)
            return {'data': data}, {'target': target}, {'idx': idx}

        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None,
        ) -> Optional['html.Div']:
            """Return None to use default display functions."""
            return None

    return _NonDisplayDataset(split='test', device=torch.device('cpu'))


# Backend Tests - get_dataset_type functionality


def test_get_dataset_type_with_base_classes(
    backend,
    real_semseg_dataset,
    real_2dcd_dataset,
    synthetic_3dcd_dataset,
    real_pcr_dataset,
):
    """Test that datasets inheriting from base display classes return correct types."""
    test_cases = [
        (real_semseg_dataset, 'semseg', 'COCOStuff164KDataset'),
        (real_2dcd_dataset, '2dcd', 'LevirCdDataset'),
        (synthetic_3dcd_dataset, '3dcd', 'Synthetic3DCDDataset'),
        (real_pcr_dataset, 'pcr', 'KITTIDataset'),
    ]

    for dataset, expected_type, dataset_class_name in test_cases:
        # Store dataset in backend
        dataset_name = f"test/{dataset_class_name}"
        backend._datasets[dataset_name] = dataset

        # Test get_dataset_type returns correct type
        result_type = backend.get_dataset_type(dataset_name)
        assert (
            result_type == expected_type
        ), f"Expected type '{expected_type}' for {dataset_class_name}, got '{result_type}'"


def test_get_dataset_type_inheritance_error(backend, non_display_dataset):
    """Test that datasets not inheriting from base classes raise proper error."""
    # Store dataset in backend
    dataset_name = "test/NonDisplayDataset"
    backend._datasets[dataset_name] = non_display_dataset

    # Test that get_dataset_type raises ValueError with descriptive message
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_type(dataset_name)

    error_msg = str(exc_info.value)
    assert "must inherit from one of the base display classes" in error_msg
    assert (
        "Base2DCDDataset, Base3DCDDataset, BasePCRDataset, or BaseSemsegDataset"
        in error_msg
    )
    assert "NonDisplayDataset" in error_msg


def test_get_dataset_type_dataset_not_loaded(backend):
    """Test that get_dataset_type raises error for unloaded dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_type("nonexistent/dataset")

    assert "Dataset not loaded: nonexistent/dataset" in str(exc_info.value)


# Backend Tests - get_dataset_instance functionality


def test_get_dataset_instance(backend, real_semseg_dataset):
    """Test the get_dataset_instance helper method."""
    # Store dataset in backend
    dataset_name = "test/COCOStuff164KDataset"
    backend._datasets[dataset_name] = real_semseg_dataset

    # Test get_dataset_instance returns the correct dataset
    retrieved_dataset = backend.get_dataset_instance(dataset_name)
    assert (
        retrieved_dataset is real_semseg_dataset
    ), "get_dataset_instance should return the exact same dataset instance"
    assert isinstance(
        retrieved_dataset, COCOStuff164KDataset
    ), "Retrieved dataset should be of correct type"


def test_get_dataset_instance_not_loaded(backend):
    """Test get_dataset_instance raises error for unloaded dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_instance("nonexistent/dataset")

    assert "Dataset not loaded: nonexistent/dataset" in str(exc_info.value)


def test_get_dataset_instance_input_validation(backend):
    """Test get_dataset_instance validates input parameters."""
    # Test non-string input
    with pytest.raises(AssertionError) as exc_info:
        backend.get_dataset_instance(123)

    assert "dataset_name must be str" in str(exc_info.value)


# Integration Tests - Multiple datasets


def test_multiple_dataset_types(
    backend, real_semseg_dataset, real_2dcd_dataset, real_pcr_dataset
):
    """Test that backend can handle multiple dataset types simultaneously."""
    datasets_and_types = [
        (real_semseg_dataset, 'semseg', 'COCOStuff164KDataset', COCOStuff164KDataset),
        (real_2dcd_dataset, '2dcd', 'LevirCdDataset', LevirCdDataset),
        (real_pcr_dataset, 'pcr', 'KITTIDataset', KITTIDataset),
    ]

    # Load all datasets
    for (
        dataset,
        expected_type,
        dataset_name_suffix,
        dataset_class,
    ) in datasets_and_types:
        dataset_name = f"test/{dataset_name_suffix}"
        backend._datasets[dataset_name] = dataset

    # Verify all dataset types are detected correctly
    for (
        dataset,
        expected_type,
        dataset_name_suffix,
        dataset_class,
    ) in datasets_and_types:
        dataset_name = f"test/{dataset_name_suffix}"

        # Test type detection
        result_type = backend.get_dataset_type(dataset_name)
        assert (
            result_type == expected_type
        ), f"Type detection failed for {dataset_name_suffix}"

        # Test instance retrieval
        retrieved_dataset = backend.get_dataset_instance(dataset_name)
        assert isinstance(
            retrieved_dataset, dataset_class
        ), f"Instance retrieval failed for {dataset_name_suffix}"


def test_dataset_inheritance_hierarchy(backend):
    """Test that inheritance hierarchy is correctly detected for complex inheritance."""

    # Create a dataset that inherits from BaseSemsegDataset through another class
    class IntermediateSemsegDataset(BaseSemsegDataset):
        pass

    class FinalSemsegDataset(IntermediateSemsegDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['image']
        LABEL_NAMES = ['label']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}, {'id': 1}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ]:
            image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            label = torch.randint(0, 5, (32, 32), dtype=torch.long)
            return {'image': image}, {'label': label}, {'idx': idx}

    # Test that inheritance through multiple levels is detected
    dataset = FinalSemsegDataset(split='test', device=torch.device('cpu'))
    dataset_name = "test/FinalSemsegDataset"
    backend._datasets[dataset_name] = dataset

    result_type = backend.get_dataset_type(dataset_name)
    assert (
        result_type == 'semseg'
    ), f"Expected 'semseg' for multi-level inheritance, got '{result_type}'"
