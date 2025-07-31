"""Shared fixtures and helper functions for slpccd_dataset tests."""

import pytest
from data.datasets.change_detection_datasets.bi_temporal.slpccd_dataset import SLPCCDDataset


@pytest.fixture
def slpccd_data_root():
    """Fixture that returns the real SLPCCD dataset path."""
    return "./data/datasets/soft_links/SLPCCD"


@pytest.fixture
def slpccd_dataset_train(slpccd_data_root):
    """Fixture for creating an SLPCCDDataset instance with train split."""
    return SLPCCDDataset(
        data_root=slpccd_data_root,
        split='train',
        num_points=8192,
        random_subsample=True,
        use_hierarchy=True,
        hierarchy_levels=3,
        knn_size=16,
        cross_knn_size=16
    )


@pytest.fixture
def dataset(request, slpccd_data_root):
    """Fixture for creating an SLPCCDDataset instance with parameterized split."""
    split = request.param
    return SLPCCDDataset(
        data_root=slpccd_data_root,
        split=split,
        num_points=256,  # Use fewer points for faster testing
        use_hierarchy=False,  # Disable hierarchy for faster testing
        random_subsample=True
    )
