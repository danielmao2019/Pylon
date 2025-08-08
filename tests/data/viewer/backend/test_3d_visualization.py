"""Tests for ViewerBackend 3D visualization detection functionality.

This module tests 3D visualization requirements detection and 
REQUIRES_3D_CLASSES functionality using real dataset classes.
"""

import os
from typing import Dict, Any, Tuple

import pytest
import torch

from data.datasets.base_dataset import BaseDataset
from data.datasets.pcr_datasets.kitti_dataset import KITTIDataset
from data.datasets.semantic_segmentation_datasets.coco_stuff_164k_dataset import COCOStuff164KDataset
from data.viewer.backend.backend import ViewerBackend, REQUIRES_3D_CLASSES


@pytest.fixture
def backend():
    """Create a ViewerBackend instance for testing."""
    return ViewerBackend()


@pytest.fixture
def kitti_dataset():
    """Create a KITTI dataset instance. Ensures real data exists and fails fast if not."""
    data_root = "./data/datasets/soft_links/KITTI"
    
    # NEVER use synthetic data - force real data or fail
    assert os.path.exists(data_root), f"KITTI dataset MUST exist at {data_root}. Please set up the dataset symlink."
    
    return KITTIDataset(
        data_root=data_root,
        split="val",
        device=torch.device("cpu")
    )


@pytest.fixture  
def random_dataset():
    """Create a BaseRandomDataset for testing."""
    from data.datasets.random_datasets.base_random_dataset import BaseRandomDataset
    
    # Simple generator configuration
    gen_func_config = {
        'inputs': {
            'image': (lambda **kwargs: torch.randn(3, 224, 224, dtype=torch.float32), {})
        },
        'labels': {
            'label': (lambda **kwargs: torch.randint(0, 10, (1,), dtype=torch.int64), {})
        }
    }
    
    dataset = BaseRandomDataset(
        num_examples=5,
        gen_func_config=gen_func_config,
        split="all",
        device=torch.device("cpu")
    )
    return dataset


@pytest.fixture
def coco_stuff_dataset():
    """Create a COCO Stuff dataset instance. Ensures real data exists and fails fast if not."""
    data_root = "./data/datasets/soft_links/COCOStuff164K"
    
    # NEVER use synthetic data - force real data or fail
    assert os.path.exists(data_root), f"COCOStuff164K dataset MUST exist at {data_root}. Please set up the dataset symlink."
    
    return COCOStuff164KDataset(
        data_root=data_root,
        split="val2017",
        device=torch.device("cpu")
    )


def test_requires_3d_classes_constant_structure():
    """Test REQUIRES_3D_CLASSES constant has expected structure."""
    assert isinstance(REQUIRES_3D_CLASSES, list)
    assert len(REQUIRES_3D_CLASSES) > 0, "Should contain at least some 3D dataset classes"
    
    # Test that all entries are strings
    for class_name in REQUIRES_3D_CLASSES:
        assert isinstance(class_name, str), f"Class name {class_name} should be string"
        assert len(class_name) > 0, f"Class name {class_name} should not be empty"


def test_requires_3d_classes_contains_expected_classes():
    """Test that REQUIRES_3D_CLASSES contains expected 3D dataset classes."""
    expected_3d_classes = [
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
    ]
    
    for expected_class in expected_3d_classes:
        assert expected_class in REQUIRES_3D_CLASSES, f"{expected_class} should be in REQUIRES_3D_CLASSES"


def test_requires_3d_visualization_with_kitti_dataset(backend, kitti_dataset):
    """Test _requires_3d_visualization returns True for KITTI dataset."""
    # KITTI is a 3D dataset
    class_name = type(kitti_dataset).__name__
    assert class_name == "KITTIDataset"
    
    # Should require 3D visualization
    requires_3d = backend._requires_3d_visualization(kitti_dataset)
    assert requires_3d is True, "KITTIDataset should require 3D visualization"


def test_requires_3d_visualization_with_random_dataset(backend, random_dataset):
    """Test _requires_3d_visualization with BaseRandomDataset."""
    # BaseRandomDataset is a general dataset type
    class_name = type(random_dataset).__name__
    assert class_name == "BaseRandomDataset"
    
    # Check if it's in the hardcoded list
    requires_3d = backend._requires_3d_visualization(random_dataset)
    # BaseRandomDataset is not in REQUIRES_3D_CLASSES
    assert requires_3d is False, "BaseRandomDataset not in hardcoded list"


def test_requires_3d_visualization_with_2d_dataset(backend, coco_stuff_dataset):
    """Test _requires_3d_visualization returns False for 2D datasets."""
    # COCO Stuff is a 2D semantic segmentation dataset
    class_name = type(coco_stuff_dataset).__name__
    assert class_name == "COCOStuff164KDataset"
    
    # Should NOT require 3D visualization
    requires_3d = backend._requires_3d_visualization(coco_stuff_dataset)
    assert requires_3d is False, "COCOStuff164KDataset should not require 3D visualization"


def test_requires_3d_visualization_exact_name_matching(backend):
    """Test that 3D visualization detection requires exact class name matching."""
    
    # Create a custom dataset that inherits from BaseDataset
    class CustomDataset(BaseDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["data"]
        LABEL_NAMES = ["label"]

        def _init_annotations(self) -> None:
            self.annotations = [{"id": 0}]

        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            data = torch.randn(10, dtype=torch.float32)
            label = torch.tensor(1, dtype=torch.int64)
            return {"data": data}, {"label": label}, {"sample_idx": idx}
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            """Display method."""
            return None
    
    dataset = CustomDataset(split="test", device=torch.device("cpu"))
    
    # CustomDataset is not in REQUIRES_3D_CLASSES
    requires_3d = backend._requires_3d_visualization(dataset)
    assert requires_3d is False, "CustomDataset should not require 3D visualization"


@pytest.mark.parametrize("class_name", [
    'Base3DCDDataset', 'BasePCRDataset', 'Buffer3DDataset',
    'KITTIDataset', 'ThreeDMatchDataset', 'URB3DCDDataset'
])
def test_requires_3d_classes_individual_entries(class_name):
    """Test that individual entries in REQUIRES_3D_CLASSES are valid."""
    assert class_name in REQUIRES_3D_CLASSES, f"{class_name} should be in REQUIRES_3D_CLASSES"
    assert isinstance(class_name, str), f"{class_name} should be string"
    assert class_name.endswith('Dataset'), f"{class_name} should end with 'Dataset'"


def test_3d_detection_consistency_with_dataset_groups():
    """Test that 3D detection is consistent with dataset categorization."""
    from data.viewer.backend.backend import DATASET_GROUPS
    
    # 3D dataset groups
    groups_3d = ['3dcd', 'pcr']
    # 2D dataset groups  
    groups_2d = ['semseg', '2dcd', 'mtl', 'general']
    
    # Test that groups are properly categorized
    all_groups = set(groups_3d + groups_2d)
    dataset_group_keys = set(DATASET_GROUPS.keys())
    
    # All dataset group keys should be in our categorization
    for group_key in dataset_group_keys:
        assert group_key in all_groups, f"Dataset group {group_key} should be categorized as 2D or 3D"


def test_requires_3d_visualization_with_base_classes(backend):
    """Test behavior with base dataset classes."""
    # Since we're testing the actual function behavior, we need to understand
    # that it checks type(dataset).__name__ against REQUIRES_3D_CLASSES
    
    # Create a minimal dataset for testing
    class TestDataset(BaseDataset):
        SPLIT_OPTIONS = ["test"]
        INPUT_NAMES = ["data"]
        LABEL_NAMES = ["label"]

        def _init_annotations(self) -> None:
            self.annotations = []

        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            raise IndexError("Empty dataset")
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            return None
    
    dataset = TestDataset(split="test", device=torch.device("cpu"))
    
    # TestDataset is not in REQUIRES_3D_CLASSES
    requires_3d = backend._requires_3d_visualization(dataset)
    assert requires_3d is False


# ============================================================================
# EDGE CASES
# ============================================================================

def test_requires_3d_visualization_with_none_dataset(backend):
    """Test _requires_3d_visualization with None dataset."""
    # None has type 'NoneType', so this should return False (not in REQUIRES_3D_CLASSES)
    result = backend._requires_3d_visualization(None)
    assert result is False


def test_requires_3d_visualization_with_invalid_object(backend):
    """Test _requires_3d_visualization with non-dataset object."""
    # These should return False since their type names are not in REQUIRES_3D_CLASSES
    result1 = backend._requires_3d_visualization("not_a_dataset")
    assert result1 is False  # type(str).__name__ = 'str', not in REQUIRES_3D_CLASSES
    
    result2 = backend._requires_3d_visualization(123)
    assert result2 is False  # type(int).__name__ = 'int', not in REQUIRES_3D_CLASSES
    
    result3 = backend._requires_3d_visualization([])
    assert result3 is False  # type(list).__name__ = 'list', not in REQUIRES_3D_CLASSES
    
    result4 = backend._requires_3d_visualization({})
    assert result4 is False  # type(dict).__name__ = 'dict', not in REQUIRES_3D_CLASSES
