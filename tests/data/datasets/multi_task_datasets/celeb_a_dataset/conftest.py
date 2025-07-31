"""Shared fixtures and helper functions for CelebADataset tests."""

import pytest


@pytest.fixture
def celeb_a_data_root():
    """Fixture that provides real CelebA data path."""
    return './data/datasets/soft_links/celeb-a'
