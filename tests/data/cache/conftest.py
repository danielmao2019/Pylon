import pytest
import torch
import psutil
import copy
from typing import Any, Dict, Tuple, Optional, List
from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise


@pytest.fixture
def tensor_params():
    """Parameters for creating test tensors."""
    return {
        'dim': 1024,
        'channels': 3,
        'dtype': torch.float32
    }


@pytest.fixture
def sample_tensor(tensor_params):
    """Create a sample tensor for testing."""
    return torch.randn(
        tensor_params['channels'],
        tensor_params['dim'],
        tensor_params['dim'],
        dtype=tensor_params['dtype']
    )


@pytest.fixture
def sample_datapoint(sample_tensor):
    """Create a sample datapoint with tensor, label, and metadata."""
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([0])},
        'meta_info': {'filename': 'test.jpg'}
    }


@pytest.fixture
def three_item_cache(sample_datapoint):
    """Create a cache configured to hold exactly 3 items."""
    # Calculate memory needed for 3 items
    item_memory = CPUDatasetCache._calculate_item_memory(sample_datapoint)
    total_memory_needed = item_memory * 3

    # Set percentage to allow exactly 3 items
    max_percent = (total_memory_needed / psutil.virtual_memory().total) * 100

    return CPUDatasetCache(max_memory_percent=max_percent)


@pytest.fixture
def cache_with_items(three_item_cache, sample_datapoint):
    """Create a cache pre-populated with 3 items.

    Returns a new copy of the cache to ensure test isolation.
    """
    cache = copy.deepcopy(three_item_cache)
    for i in range(3):
        cache.put(i, copy.deepcopy(sample_datapoint))
    return cache


@pytest.fixture
def make_datapoint(tensor_params):
    """Factory fixture to create datapoints with unique tensors and metadata."""
    def _make_datapoint(index: int):
        tensor = torch.randn(
            tensor_params['channels'],
            tensor_params['dim'],
            tensor_params['dim'],
            dtype=tensor_params['dtype']
        )
        return {
            'inputs': {'image': tensor},
            'labels': {'class': torch.tensor([index])},
            'meta_info': {'filename': f'test_{index}.jpg'}
        }
    return _make_datapoint


@pytest.fixture
def SampleDataset():
    """A minimal dataset implementation for testing BaseDataset functionality."""
    class _SampleDataset(BaseDataset):
        SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
        DATASET_SIZE = {'train': 80, 'val': 10, 'test': 10, 'weird': 0}  # Class attribute for predefined splits
        INPUT_NAMES = ['input']
        LABEL_NAMES = ['label']

        def __init__(self, data_root=None, split=None, split_percentages=None, **kwargs):
            # This dataset has predefined splits, so split=None is not allowed
            if split is None:
                raise ValueError("SampleDataset has predefined splits - split=None is not allowed. Use SampleDatasetWithoutPredefinedSplits instead.")
            
            super().__init__(data_root=data_root, split=split, split_percentages=split_percentages, **kwargs)

        def _init_annotations(self) -> None:
            # BaseDataset split logic:
            # 1. If split_percentages provided: load ALL data, BaseDataset will apply split after
            # 2. If no split_percentages: load only the specific split's data
            
            if hasattr(self, 'split_percentages') and self.split_percentages is not None:
                # Load ALL data - BaseDataset will apply percentage split after this
                self.annotations = list(range(100))  # 0-99
            elif self.split == 'train':
                self.annotations = list(range(80))  # 0-79
            elif self.split == 'val':
                self.annotations = list(range(80, 90))  # 80-89
            elif self.split == 'test':
                self.annotations = list(range(90, 100))  # 90-99
            elif self.split == 'weird':
                self.annotations = []  # Empty split
            else:
                raise ValueError(f"Unknown split: {self.split}")

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            # Create tensors on CPU initially - BaseDataset handles device transfer
            annotation_value = self.annotations[idx]
            return {
                'input': torch.tensor(annotation_value, dtype=torch.float32),
            }, {
                'label': torch.tensor(annotation_value, dtype=torch.float32),
            }, {
            }

        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None
        ) -> Optional['html.Div']:
            """Return None to use default display functions."""
            return None

    return _SampleDataset


@pytest.fixture
def SampleDatasetWithoutPredefinedSplits():
    """A dataset implementation without predefined splits for testing split_percentages functionality."""
    class _SampleDatasetWithoutPredefinedSplits(BaseDataset):
        SPLIT_OPTIONS = ['train', 'val', 'test', 'weird']
        DATASET_SIZE = 100  # Total size for percentage-based splitting
        INPUT_NAMES = ['input']
        LABEL_NAMES = ['label']

        def _init_annotations(self) -> None:
            # Always load everything - splitting is handled by BaseDataset
            self.annotations = list(range(100))  # 0-99

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            # Create tensors on CPU initially - BaseDataset handles device transfer
            annotation_value = self.annotations[idx]
            return {
                'input': torch.tensor(annotation_value, dtype=torch.float32),
            }, {
                'label': torch.tensor(annotation_value, dtype=torch.float32),
            }, {
            }

        @staticmethod
        def display_datapoint(
            datapoint: Dict[str, Any],
            class_labels: Optional[Dict[str, List[str]]] = None,
            camera_state: Optional[Dict[str, Any]] = None,
            settings_3d: Optional[Dict[str, Any]] = None
        ) -> Optional['html.Div']:
            """Return None to use default display functions."""
            return None

    return _SampleDatasetWithoutPredefinedSplits


@pytest.fixture
def random_transforms():
    """Random transforms for testing cache behavior with randomization."""
    noise = RandomNoise(std=0.2)  # Add significant noise for visible effect

    return {
        'class': Compose,
        'args': {
            'transforms': [
                (noise, [('inputs', 'input')]),
            ]
        }
    }


@pytest.fixture
def TestDatasetWithDataRoot():
    """Fixture that provides test dataset class that has a data_root for cache directory testing."""
    class TestDatasetWithDataRootImpl(BaseDataset):
        SPLIT_OPTIONS = ['train', 'test']
        DATASET_SIZE = {'train': 5, 'test': 3}
        INPUT_NAMES = ['data']
        LABEL_NAMES = ['target']
        SHA1SUM = None
        
        def _init_annotations(self) -> None:
            """Initialize with dummy annotations."""
            # DATASET_SIZE is normalized to int in _init_split, so use it directly
            self.annotations = [{'index': i} for i in range(self.DATASET_SIZE)]
        
        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            """Load dummy datapoint."""
            inputs = {'data': torch.randn(3, 32, 32)}
            labels = {'target': torch.tensor(idx % 2)}
            meta_info = {'annotation_idx': idx}
            return inputs, labels, meta_info
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            return None
    
    return TestDatasetWithDataRootImpl


@pytest.fixture
def TestDatasetWithoutDataRoot():
    """Fixture that provides test dataset class that does NOT have a data_root for cache directory testing."""
    class TestDatasetWithoutDataRootImpl(BaseDataset):
        SPLIT_OPTIONS = ['train', 'test']
        DATASET_SIZE = {'train': 5, 'test': 3}
        INPUT_NAMES = ['data']
        LABEL_NAMES = ['target']
        SHA1SUM = None

        def _init_annotations(self) -> None:
            """Initialize with dummy annotations."""
            # DATASET_SIZE is normalized to int in _init_split, so use it directly
            self.annotations = [{'index': i} for i in range(self.DATASET_SIZE)]
        
        def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
            """Load dummy datapoint."""
            inputs = {'data': torch.randn(3, 32, 32)}
            labels = {'target': torch.tensor(idx % 2)}
            meta_info = {'annotation_idx': idx}
            return inputs, labels, meta_info
        
        @staticmethod
        def display_datapoint(datapoint, class_labels=None, camera_state=None, settings_3d=None):
            return None
    
    return TestDatasetWithoutDataRootImpl
