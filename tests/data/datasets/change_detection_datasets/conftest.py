"""Shared fixtures for change detection dataset tests.

This module consolidates common data root fixtures for change detection datasets
to eliminate duplication across multiple test files.
"""

import pytest


# =============================================================================
# Data Root Fixtures - Bi-temporal Change Detection Datasets
# =============================================================================

@pytest.fixture
def air_change_data_root():
    """Fixture that provides real AirChange data path."""
    return "./data/datasets/soft_links/AirChange"


@pytest.fixture
def cdd_data_root():
    """Fixture that returns the real CDD dataset path."""
    return "./data/datasets/soft_links/CDD"


@pytest.fixture
def kc_3d_data_root():
    """Fixture that returns the real KC-3D dataset path."""
    return "./data/datasets/soft_links/KC-3D"


@pytest.fixture
def levir_cd_data_root():
    """Fixture that returns the real LEVIR-CD dataset path."""
    return "./data/datasets/soft_links/LEVIR-CD"


@pytest.fixture
def oscd_data_root():
    """Fixture that returns the real OSCD dataset path."""
    return "./data/datasets/soft_links/OSCD"


@pytest.fixture
def slpccd_data_root():
    """Fixture that returns the real SLPCCD dataset path."""
    return "./data/datasets/soft_links/SLPCCD"


@pytest.fixture
def sysu_cd_data_root():
    """Fixture that returns the real SYSU-CD dataset path."""
    return "./data/datasets/soft_links/SYSU-CD"


@pytest.fixture
def urb3dcd_data_root():
    """Fixture that returns the real Urb3DCD dataset path."""
    return "./data/datasets/soft_links/Urb3DCD"


@pytest.fixture
def xview2_data_root():
    """Fixture that returns the real xView2 dataset path."""
    return "./data/datasets/soft_links/xView2"


# =============================================================================
# Data Root Fixtures - Single-temporal Change Detection Datasets
# =============================================================================

@pytest.fixture
def ppsl_data_root():
    """Fixture that returns the real PPSL dataset path."""
    return "./data/datasets/soft_links/PPSL"


@pytest.fixture
def i3pe_data_root():
    """Fixture that returns the real I3PE dataset path."""
    return "./data/datasets/soft_links/I3PE"


# =============================================================================
# Data Root Factory Fixture
# =============================================================================

@pytest.fixture
def get_change_detection_data_root():
    """Factory fixture for getting change detection dataset data roots.
    
    Returns:
        Function that takes dataset name and returns corresponding data root path.
    """
    def _get_data_root(dataset_name: str) -> str:
        """Get data root path for a change detection dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'air_change', 'cdd', etc.)
            
        Returns:
            Data root path for the specified dataset.
            
        Raises:
            ValueError: If dataset name is not recognized.
        """
        dataset_paths = {
            'air_change': './data/datasets/soft_links/AirChange',
            'cdd': './data/datasets/soft_links/CDD',
            'kc_3d': './data/datasets/soft_links/KC-3D',
            'levir_cd': './data/datasets/soft_links/LEVIR-CD',
            'oscd': './data/datasets/soft_links/OSCD',
            'slpccd': './data/datasets/soft_links/SLPCCD',
            'sysu_cd': './data/datasets/soft_links/SYSU-CD',
            'urb3dcd': './data/datasets/soft_links/Urb3DCD',
            'xview2': './data/datasets/soft_links/xView2',
            'ppsl': './data/datasets/soft_links/PPSL',
            'i3pe': './data/datasets/soft_links/I3PE',
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown change detection dataset: {dataset_name}. "
                           f"Available datasets: {list(dataset_paths.keys())}")
        
        return dataset_paths[dataset_name]
    
    return _get_data_root
