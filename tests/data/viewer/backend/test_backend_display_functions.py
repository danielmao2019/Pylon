"""Tests for backend display functions.

This module tests the critical backend functionality for determining dataset types
and managing dataset instances for the display system.
"""
import pytest
import torch
from typing import Dict, Any, Tuple, Optional, List
from data.datasets.base_dataset import BaseDataset
from data.datasets.semantic_segmentation_datasets.base_semseg_dataset import BaseSemsegDataset
from data.datasets.change_detection_datasets.base_2d_cd_dataset import Base2DCDDataset
from data.datasets.change_detection_datasets.base_3d_cd_dataset import Base3DCDDataset
from data.datasets.pcr_datasets.base_pcr_dataset import BasePCRDataset
from data.viewer.backend.backend import ViewerBackend


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def MockSemsegDataset():
    """Mock semantic segmentation dataset for testing."""
    class _MockSemsegDataset(BaseSemsegDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['image']
        LABEL_NAMES = ['label']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            label = torch.randint(0, 5, (32, 32), dtype=torch.long)
            return {'image': image}, {'label': label}, {'idx': idx}

    return _MockSemsegDataset


@pytest.fixture
def Mock2DCDDataset():
    """Mock 2D change detection dataset for testing."""
    class _Mock2DCDDataset(Base2DCDDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['img_1', 'img_2']
        LABEL_NAMES = ['change_map']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            img1 = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            img2 = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            change_map = torch.randint(0, 2, (32, 32), dtype=torch.long)
            return {'img_1': img1, 'img_2': img2}, {'change_map': change_map}, {'idx': idx}

    return _Mock2DCDDataset


@pytest.fixture
def Mock3DCDDataset():
    """Mock 3D change detection dataset for testing."""
    class _Mock3DCDDataset(Base3DCDDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['pc_1', 'pc_2']
        LABEL_NAMES = ['change_map']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            n_points = 1000
            pc1 = {'pos': torch.randn(n_points, 3, dtype=torch.float32)}
            pc2 = {'pos': torch.randn(n_points, 3, dtype=torch.float32)}
            change_map = torch.randint(0, 2, (n_points,), dtype=torch.long)
            return {'pc_1': pc1, 'pc_2': pc2}, {'change_map': change_map}, {'idx': idx}

    return _Mock3DCDDataset


@pytest.fixture
def MockPCRDataset():
    """Mock point cloud registration dataset for testing."""
    class _MockPCRDataset(BasePCRDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['src_pc', 'tgt_pc']
        LABEL_NAMES = ['transform']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            n_points = 1000
            src_pc = {'pos': torch.randn(n_points, 3, dtype=torch.float32)}
            tgt_pc = {'pos': torch.randn(n_points, 3, dtype=torch.float32)}
            transform = torch.eye(4, dtype=torch.float32)
            return {'src_pc': src_pc, 'tgt_pc': tgt_pc}, {'transform': transform}, {'idx': idx}

    return _MockPCRDataset


@pytest.fixture
def MockNonDisplayDataset():
    """Mock dataset that doesn't inherit from any base display class."""
    class _MockNonDisplayDataset(BaseDataset):
        SPLIT_OPTIONS = ['test']
        INPUT_NAMES = ['data']
        LABEL_NAMES = ['target']

        def _init_annotations(self) -> None:
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            data = torch.randn(10, dtype=torch.float32)
            target = torch.tensor(1, dtype=torch.long)
            return {'data': data}, {'target': target}, {'idx': idx}

        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None
        ) -> Optional['html.Div']:
            """Return None to use default display functions."""
            return None

    return _MockNonDisplayDataset


# Backend Tests - get_dataset_type functionality


def test_get_dataset_type_with_base_classes(backend, MockSemsegDataset, Mock2DCDDataset, Mock3DCDDataset, MockPCRDataset):
    """Test that datasets inheriting from base display classes return correct types."""
    test_cases = [
        (MockSemsegDataset, 'semseg'),
        (Mock2DCDDataset, '2dcd'),
        (Mock3DCDDataset, '3dcd'),
        (MockPCRDataset, 'pcr'),
    ]
    
    for dataset_class, expected_type in test_cases:
        # Create and store dataset in backend
        dataset = dataset_class(split='test', device=torch.device('cpu'))
        dataset_name = f"test/{dataset_class.__name__}"
        backend._datasets[dataset_name] = dataset
        
        # Test get_dataset_type returns correct type
        result_type = backend.get_dataset_type(dataset_name)
        assert result_type == expected_type, f"Expected type '{expected_type}' for {dataset_class.__name__}, got '{result_type}'"


def test_get_dataset_type_inheritance_error(backend, MockNonDisplayDataset):
    """Test that datasets not inheriting from base classes raise proper error."""
    # Create and store dataset in backend
    dataset = MockNonDisplayDataset(split='test', device=torch.device('cpu'))
    dataset_name = "test/MockNonDisplayDataset"
    backend._datasets[dataset_name] = dataset
    
    # Test that get_dataset_type raises ValueError with descriptive message
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_type(dataset_name)
    
    error_msg = str(exc_info.value)
    assert "must inherit from one of the base display classes" in error_msg
    assert "Base2DCDDataset, Base3DCDDataset, BasePCRDataset, or BaseSemsegDataset" in error_msg
    assert "MockNonDisplayDataset" in error_msg


def test_get_dataset_type_dataset_not_loaded(backend):
    """Test that get_dataset_type raises error for unloaded dataset."""
    with pytest.raises(ValueError) as exc_info:
        backend.get_dataset_type("nonexistent/dataset")
    
    assert "Dataset not loaded: nonexistent/dataset" in str(exc_info.value)


# Backend Tests - get_dataset_instance functionality


def test_get_dataset_instance(backend, MockSemsegDataset):
    """Test the get_dataset_instance helper method."""
    # Create and store dataset in backend
    dataset = MockSemsegDataset(split='test', device=torch.device('cpu'))
    dataset_name = "test/MockSemsegDataset"
    backend._datasets[dataset_name] = dataset
    
    # Test get_dataset_instance returns the correct dataset
    retrieved_dataset = backend.get_dataset_instance(dataset_name)
    assert retrieved_dataset is dataset, "get_dataset_instance should return the exact same dataset instance"
    assert isinstance(retrieved_dataset, MockSemsegDataset), "Retrieved dataset should be of correct type"


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


def test_multiple_dataset_types(backend, MockSemsegDataset, Mock2DCDDataset, MockPCRDataset):
    """Test that backend can handle multiple dataset types simultaneously."""
    datasets_and_types = [
        (MockSemsegDataset, 'semseg'),
        (Mock2DCDDataset, '2dcd'), 
        (MockPCRDataset, 'pcr'),
    ]
    
    # Load all datasets
    for dataset_class, expected_type in datasets_and_types:
        dataset = dataset_class(split='test', device=torch.device('cpu'))
        dataset_name = f"test/{dataset_class.__name__}"
        backend._datasets[dataset_name] = dataset
    
    # Verify all dataset types are detected correctly
    for dataset_class, expected_type in datasets_and_types:
        dataset_name = f"test/{dataset_class.__name__}"
        
        # Test type detection
        result_type = backend.get_dataset_type(dataset_name)
        assert result_type == expected_type, f"Type detection failed for {dataset_class.__name__}"
        
        # Test instance retrieval
        retrieved_dataset = backend.get_dataset_instance(dataset_name)
        assert isinstance(retrieved_dataset, dataset_class), f"Instance retrieval failed for {dataset_class.__name__}"


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
            self.annotations = [{'id': 0}]

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
            label = torch.randint(0, 5, (32, 32), dtype=torch.long)
            return {'image': image}, {'label': label}, {'idx': idx}
    
    # Test that inheritance through multiple levels is detected
    dataset = FinalSemsegDataset(split='test', device=torch.device('cpu'))
    dataset_name = "test/FinalSemsegDataset"
    backend._datasets[dataset_name] = dataset
    
    result_type = backend.get_dataset_type(dataset_name)
    assert result_type == 'semseg', f"Expected 'semseg' for multi-level inheritance, got '{result_type}'"