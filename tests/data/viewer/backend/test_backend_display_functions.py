"""Tests for backend display functions.

This module tests the critical backend functionality for managing dataset instances
for the display system using real datasets.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from data.datasets.base_dataset import BaseDataset
from data.datasets.change_detection_datasets.bi_temporal.levir_cd_dataset import (
    LevirCdDataset,
)
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
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
def coco_stuff_164k_dataset():
    """COCOStuff164K dataset for testing."""
    data_root = "./data/datasets/soft_links/COCOStuff164K"

    # Assert dataset is available - fail fast if not
    assert os.path.exists(
        data_root
    ), f"COCOStuff164K dataset must be available at {data_root}"

    # Create dataset - let it fail if it can't be created
    dataset = COCOStuff164KDataset(
        data_root=data_root,
        split="val2017",  # Use smaller validation split for testing
        device=torch.device("cpu"),
    )
    return dataset


@pytest.fixture
def levir_cd_dataset():
    """LEVIR-CD dataset for testing."""
    data_root = "./data/datasets/soft_links/LEVIR-CD"

    # Assert dataset is available - fail fast if not
    assert os.path.exists(
        data_root
    ), f"LEVIR-CD dataset must be available at {data_root}"

    # Create dataset - let it fail if it can't be created
    dataset = LevirCdDataset(
        data_root=data_root,
        split="test",  # Use test split for testing
        device=torch.device("cpu"),
    )
    return dataset


@pytest.fixture
def kitti_dataset():
    """KITTI dataset for testing."""
    data_root = "./data/datasets/soft_links/KITTI"

    # Assert dataset is available - fail fast if not
    assert os.path.exists(data_root), f"KITTI dataset must be available at {data_root}"

    # Create dataset - let it fail if it can't be created
    dataset = KITTIDataset(
        data_root=data_root,
        split="val",  # Use validation split for testing
        device=torch.device("cpu"),
    )
    return dataset


@pytest.fixture
def non_display_dataset():
    """Dataset that doesn't inherit from any base display class for general testing."""

    class _NonDisplayDataset(BaseDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["data"]
        LABEL_NAMES = ["target"]

        def _init_annotations(self) -> None:
            self.annotations = [{"id": 0}, {"id": 1}]  # Small test dataset

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ]:
            data = torch.randn(10, dtype=torch.float32)
            target = torch.tensor(1, dtype=torch.long)
            return {"data": data}, {"target": target}, {"sample_idx": idx}

        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None,
        ):
            """Return None to use default display functions."""
            # Explicitly acknowledge all parameters to avoid unused warnings
            _ = datapoint, class_labels, camera_state, settings_3d
            return None

    return _NonDisplayDataset(data_root=None, split="test", device=torch.device("cpu"))


# Backend Tests - Dataset Management Functionality


def test_dataset_instance_management_with_real_datasets(
    backend,
    coco_stuff_164k_dataset,
    levir_cd_dataset,
    kitti_dataset,
):
    """Test that backend can manage real dataset instances correctly."""
    test_cases = [
        (coco_stuff_164k_dataset, "COCOStuff164KDataset", COCOStuff164KDataset),
        (levir_cd_dataset, "LevirCdDataset", LevirCdDataset),
        (kitti_dataset, "KITTIDataset", KITTIDataset),
    ]

    for dataset, dataset_class_name, expected_type in test_cases:
        # Store dataset in backend
        dataset_name = f"test/{dataset_class_name}"
        backend._datasets[dataset_name] = dataset

        # Test get_dataset_instance returns correct dataset
        retrieved_dataset = backend.get_dataset_instance(dataset_name)
        assert (
            retrieved_dataset is dataset
        ), f"Should return the exact same instance for {dataset_class_name}"
        assert isinstance(
            retrieved_dataset, expected_type
        ), f"Should maintain correct type for {dataset_class_name}"


def test_non_display_dataset_management(backend, non_display_dataset):
    """Test that backend can manage various types of datasets including non-display ones."""
    # Store dataset in backend
    dataset_name = "test/NonDisplayDataset"
    backend._datasets[dataset_name] = non_display_dataset

    # Test that backend can manage non-display datasets
    retrieved_dataset = backend.get_dataset_instance(dataset_name)
    assert (
        retrieved_dataset is non_display_dataset
    ), "Should retrieve the exact same dataset instance"

    # Test that the dataset is functional
    assert len(retrieved_dataset) == 2, "Dataset should have 2 samples"
    sample = retrieved_dataset[0]

    # Handle both data formats
    if isinstance(sample, dict) and "inputs" in sample:
        inputs = sample["inputs"]
        labels = sample["labels"]
    else:
        inputs, labels, _ = sample

    assert (
        "data" in inputs
    ), f"Sample should have 'data' in inputs, got keys: {list(inputs.keys())}"
    assert (
        "target" in labels
    ), f"Sample should have 'target' in labels, got keys: {list(labels.keys())}"


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
    backend, coco_stuff_164k_dataset, levir_cd_dataset, kitti_dataset
):
    """Test that backend can handle multiple dataset types simultaneously."""
    datasets_and_info = [
        (coco_stuff_164k_dataset, "COCOStuff164KDataset", COCOStuff164KDataset),
        (levir_cd_dataset, "LevirCdDataset", LevirCdDataset),
        (kitti_dataset, "KITTIDataset", KITTIDataset),
    ]

    # Load all datasets
    for dataset, dataset_name_suffix, dataset_class in datasets_and_info:
        dataset_name = f"test/{dataset_name_suffix}"
        backend._datasets[dataset_name] = dataset

    # Verify all datasets are managed correctly
    for dataset, dataset_name_suffix, dataset_class in datasets_and_info:
        dataset_name = f"test/{dataset_name_suffix}"

        # Test instance retrieval
        retrieved_dataset = backend.get_dataset_instance(dataset_name)
        assert isinstance(
            retrieved_dataset, dataset_class
        ), f"Instance retrieval failed for {dataset_name_suffix}"

        # Test that the retrieved dataset is the same instance
        assert (
            retrieved_dataset is dataset
        ), f"Retrieved dataset should be the same instance for {dataset_name_suffix}"


def test_dataset_inheritance_hierarchy(backend):
    """Test that backend can manage datasets with complex inheritance hierarchies."""

    # Create a dataset that inherits from BaseSemsegDataset through another class
    class IntermediateSemsegDataset(BaseSemsegDataset):
        pass

    class FinalSemsegDataset(IntermediateSemsegDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["image"]
        LABEL_NAMES = ["label"]

        def _init_annotations(self) -> None:
            self.annotations = [{"id": 0}, {"id": 1}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ]:
            image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            label = torch.randint(0, 5, (32, 32), dtype=torch.long)
            return {"image": image}, {"label": label}, {"sample_idx": idx}

    # Test that datasets with complex inheritance can be managed by the backend
    dataset = FinalSemsegDataset(data_root=None, split="test", device=torch.device("cpu"))
    dataset_name = "test/FinalSemsegDataset"
    backend._datasets[dataset_name] = dataset

    # Test that we can retrieve the dataset instance
    retrieved_dataset = backend.get_dataset_instance(dataset_name)
    assert (
        retrieved_dataset is dataset
    ), "Should retrieve the exact same dataset instance"
    assert isinstance(
        retrieved_dataset, FinalSemsegDataset
    ), "Should maintain the correct type"


def test_dataset_functionality_with_real_data(backend, coco_stuff_164k_dataset):
    """Test that real datasets are functional through the backend."""
    # Store dataset in backend
    dataset_name = "test/COCOStuff164KDataset"
    backend._datasets[dataset_name] = coco_stuff_164k_dataset

    # Test basic dataset properties
    retrieved_dataset = backend.get_dataset_instance(dataset_name)
    assert len(retrieved_dataset) > 0, "Dataset should have samples"

    # Test that we can load a sample
    sample = retrieved_dataset[0]

    # Check if it's the new format (dict with 'inputs', 'labels', 'meta_info') or old format (tuple)
    if isinstance(sample, dict) and "inputs" in sample:
        # New format
        inputs = sample["inputs"]
        labels = sample["labels"]
        meta_info = sample["meta_info"]
    else:
        # Old format (tuple)
        assert (
            isinstance(sample, tuple) and len(sample) == 3
        ), "Sample should be a tuple with inputs, labels, meta_info"
        inputs, labels, meta_info = sample

    # Test that inputs and labels are dictionaries with expected structure
    assert isinstance(inputs, dict), "Inputs should be a dictionary"
    assert isinstance(labels, dict), "Labels should be a dictionary"
    assert isinstance(meta_info, dict), "Meta info should be a dictionary"

    # Test that we have expected keys (based on COCOStuff164K dataset structure)
    assert "image" in inputs, "Should have image in inputs"
    # COCOStuff164K uses 'label' not 'mask'
    assert "label" in labels, "Should have label in labels"
