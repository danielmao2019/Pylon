# ================================================================================
# Global Test Configuration - Centralized Dataset Paths for ALL Tests
# ================================================================================
"""
Global conftest.py for all Pylon tests.

This file provides centralized dataset root fixtures that are available to ALL test
cases across the entire test suite. This ensures consistent dataset path management
and eliminates duplication of fixture definitions.
"""

import pytest

# ================================================================================
# Dataset Root Fixtures - Centralized dataset paths for ALL tests
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
