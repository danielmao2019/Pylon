from typing import Any, Dict, Tuple, Optional, List
import pytest
import torch
import os
import glob
from data.datasets.base_dataset import BaseDataset
from data.transforms.compose import Compose
from data.transforms.random_noise import RandomNoise


def pytest_addoption(parser):
    """Add custom command line options for dataset testing."""
    parser.addoption(
        "--samples",
        action="store",
        default=None,
        type=int,
        help="Maximum number of datapoints to test per dataset (default: test full dataset)"
    )
    parser.addoption(
        "--cpu",
        action="store_true",
        default=False,
        help="Use CPU device for datasets instead of GPU (default: use GPU)"
    )


@pytest.fixture
def max_samples(request):
    """Fixture to get the maximum number of samples to test from command line."""
    return request.config.getoption("--samples")


@pytest.fixture
def use_cpu_device(request):
    """Fixture to get the CPU device flag from command line."""
    return request.config.getoption("--cpu")


@pytest.fixture
def get_samples_to_test():
    """
    Fixture that provides helper function to determine how many samples to test.
    """
    def _get_samples_to_test(dataset_length: int, max_samples: int = None) -> int:
        """
        Helper function to determine how many samples to test based on command line args.

        Args:
            dataset_length: Total number of samples in the dataset
            max_samples: Value from --samples command line argument (can be None)

        Returns:
            Number of samples to test (full dataset if --samples not provided)
        """
        if max_samples is not None:
            return min(dataset_length, max_samples)
        else:
            return dataset_length
    return _get_samples_to_test


@pytest.fixture
def get_device():
    """
    Fixture that provides helper function to determine device based on command line flag.
    """
    def _get_device(use_cpu: bool = False) -> str:
        """
        Helper function to determine device based on command line args.

        Args:
            use_cpu: Value from --cpu command line argument

        Returns:
            Device string: 'cpu' if --cpu flag is provided, 'cuda' otherwise
        """
        if use_cpu:
            return 'cpu'
        else:
            return 'cuda'
    return _get_device


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
            # Create a random tensor for testing transforms
            # Use the annotation index as the random seed for reproducibility
            torch.manual_seed(self.annotations[idx])
            tensor = torch.randn(3, 32, 32)  # Random noise
            return {'input': tensor}, {'label': self.annotations[idx]}, {}

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
        INPUT_NAMES = ['input']
        LABEL_NAMES = ['label']
        
        def __init__(self, data_root=None, split=None, split_percentages=None, **kwargs):
            # Set DATASET_SIZE only when using split_percentages (total size before splitting)
            if split_percentages is not None:
                self.DATASET_SIZE = 100  # Total size for percentage-based splitting
            # For split=None, don't set DATASET_SIZE (allows loading everything)
            
            super().__init__(data_root=data_root, split=split, split_percentages=split_percentages, **kwargs)

        def _init_annotations(self) -> None:
            # Always load everything - splitting is handled by BaseDataset or user specifies split=None
            self.annotations = list(range(100))  # 0-99

        def _load_datapoint(self, idx: int) -> Tuple[
            Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any],
        ]:
            # Create a random tensor for testing transforms
            # Use the annotation index as the random seed for reproducibility
            torch.manual_seed(self.annotations[idx])
            tensor = torch.randn(3, 32, 32)  # Random noise
            return {'input': tensor}, {'label': self.annotations[idx]}, {}

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


# ================================================================================
# Dataset Root Fixtures - Centralized dataset paths for all tests
# ================================================================================

# --------------------------------------------------------------------------------
# Change Detection Datasets (Bi-temporal)
# --------------------------------------------------------------------------------

@pytest.fixture
def air_change_data_root():
    """Fixture that provides AirChange dataset path."""
    return "./data/datasets/soft_links/AirChange"


@pytest.fixture
def cdd_data_root():
    """Fixture that provides CDD dataset path."""
    return "./data/datasets/soft_links/CDD"


@pytest.fixture
def kc_3d_data_root():
    """Fixture that provides KC-3D dataset path."""
    return "./data/datasets/soft_links/KC-3D"


@pytest.fixture
def levir_cd_data_root():
    """Fixture that provides LEVIR-CD dataset path."""
    return "./data/datasets/soft_links/LEVIR-CD"


@pytest.fixture
def oscd_data_root():
    """Fixture that provides OSCD dataset path."""
    return "./data/datasets/soft_links/OSCD"


@pytest.fixture
def slpccd_data_root():
    """Fixture that provides SLPCCD dataset path."""
    return "./data/datasets/soft_links/SLPCCD"


@pytest.fixture
def sysu_cd_data_root():
    """Fixture that provides SYSU-CD dataset path."""
    return "./data/datasets/soft_links/SYSU-CD"


@pytest.fixture
def urb3dcd_data_root():
    """Fixture that provides Urb3DCD dataset path."""
    return "./data/datasets/soft_links/Urb3DCD"


@pytest.fixture
def xview2_data_root():
    """Fixture that provides xView2 dataset path."""
    return "./data/datasets/soft_links/xView2"


# --------------------------------------------------------------------------------
# Change Detection Datasets (Single-temporal)
# --------------------------------------------------------------------------------

@pytest.fixture
def ppsl_data_root():
    """Fixture that provides PPSL dataset path."""
    return "./data/datasets/soft_links/PPSL"


@pytest.fixture
def i3pe_data_root():
    """Fixture that provides I3PE dataset path."""
    return "./data/datasets/soft_links/I3PE"


# --------------------------------------------------------------------------------
# Multi-task Datasets
# --------------------------------------------------------------------------------

@pytest.fixture
def ade20k_data_root():
    """Fixture that provides ADE20K dataset path."""
    return "./data/datasets/soft_links/ADE20K"


@pytest.fixture
def celeb_a_data_root():
    """Fixture that provides CelebA dataset path."""
    return "./data/datasets/soft_links/celeb-a"


@pytest.fixture
def city_scapes_data_root():
    """Fixture that provides CityScapes dataset path."""
    return "./data/datasets/soft_links/city-scapes"


@pytest.fixture
def multi_task_facial_landmark_data_root():
    """Fixture that provides Multi-task Facial Landmark dataset path."""
    return "./data/datasets/soft_links/multi-task-facial-landmark"


@pytest.fixture
def nyu_v2_data_root():
    """Fixture that provides NYU-v2 dataset path."""
    return "./data/datasets/soft_links/NYUD_MT"


@pytest.fixture
def pascal_context_data_root():
    """Fixture that provides PASCAL Context dataset path."""
    return "./data/datasets/soft_links/PASCAL_MT"


# --------------------------------------------------------------------------------
# Point Cloud Registration (PCR) Datasets
# --------------------------------------------------------------------------------

@pytest.fixture
def kitti_data_root():
    """Fixture that provides KITTI dataset path."""
    return "./data/datasets/soft_links/KITTI"


@pytest.fixture
def modelnet40_data_root():
    """Fixture that provides ModelNet40 dataset path."""
    return "./data/datasets/soft_links/ModelNet40"


@pytest.fixture
def threedmatch_data_root():
    """Fixture that provides 3DMatch dataset path."""
    return "./data/datasets/soft_links/threedmatch"


# --------------------------------------------------------------------------------
# Semantic Segmentation Datasets
# --------------------------------------------------------------------------------

@pytest.fixture
def coco_stuff_164k_data_root():
    """Fixture that provides COCOStuff164K dataset path."""
    return "./data/datasets/soft_links/COCOStuff164K"


@pytest.fixture
def whu_bd_data_root():
    """Fixture that provides WHU-BD dataset path."""
    return "./data/datasets/soft_links/WHU-BD"


# --------------------------------------------------------------------------------
# Torchvision Datasets
# --------------------------------------------------------------------------------

@pytest.fixture
def mnist_data_root():
    """Fixture that provides MNIST data path."""
    return "./data/datasets/soft_links/MNIST"


# --------------------------------------------------------------------------------
# Cache Directory Fixtures
# --------------------------------------------------------------------------------

@pytest.fixture
def cache_dir():
    """Fixture that provides the main cache directory path."""
    return "./data/cache"


@pytest.fixture
def modelnet40_cache_file():
    """Fixture that provides ModelNet40 cache file path."""
    return "./data/datasets/soft_links/ModelNet40/../ModelNet40_cache.json"


@pytest.fixture(autouse=True)
def cleanup_test_cache_files():
    """Automatically clean up test cache files after each test."""
    yield  # Run the test first
    
    # Clean up any test cache files
    cache_dir = "./data/cache"
    if os.path.exists(cache_dir):
        test_cache_patterns = [
            "test_*.json",
            "*_test.json",
            "temp_*.json"
        ]
        
        for pattern in test_cache_patterns:
            cache_files = glob.glob(os.path.join(cache_dir, pattern))
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                except (OSError, FileNotFoundError):
                    pass  # File already removed or doesn't exist
