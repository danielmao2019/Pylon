"""
Determinism tests for PPSLDataset using TorchvisionWrapper transforms.

Tests that PPSLDataset produces deterministic results when transforms are applied during loading.
Uses real WHU_BD_Dataset as source instead of mocked data.
"""
import torch
import pytest
from data.datasets.change_detection_datasets.single_temporal.ppsl_dataset import PPSLDataset
from data.datasets.semantic_segmentation_datasets.whu_bd_dataset import WHU_BD_Dataset


@pytest.fixture(scope='module')
def whu_bd_source_dataset():
    """Create a real WHU_BD_Dataset for testing PPSLDataset determinism, following config setup."""
    # Follow the exact configuration from configs/common/datasets/change_detection/train/ppsl_whu_bd.py
    dataset = WHU_BD_Dataset(
        data_root="./data/datasets/soft_links/WHU-BD",
        split="train",
        use_cpu_cache=False,
        use_disk_cache=False
    )
    # Assert the dataset has data - fail fast if it doesn't
    assert len(dataset) > 0, "WHU_BD_Dataset is empty - no data available for testing"

    return dataset


def test_ppsl_dataset_deterministic_transforms(whu_bd_source_dataset):
    """Test that PPSLDataset applies transforms deterministically during loading."""
    # Create two PPSLDataset instances with same seed
    dataset1 = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=5,
        base_seed=123,
        use_cache=False
    )

    dataset2 = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=5,
        base_seed=123,
        use_cache=False
    )

    # Test that same indices produce identical results
    for idx in range(min(3, len(dataset1))):  # Test first 3 datapoints or all if less
        # Load datapoint from both datasets
        datapoint1 = dataset1[idx]
        datapoint2 = dataset2[idx]

        # Images should be identical (ColorJitter and RandomAffine applied deterministically)
        assert torch.allclose(datapoint1['inputs']['img_1'], datapoint2['inputs']['img_1']), \
            f"img_1 should be identical for idx {idx} (ColorJitter determinism)"

        assert torch.allclose(datapoint1['inputs']['img_2'], datapoint2['inputs']['img_2']), \
            f"img_2 should be identical for idx {idx} (RandomAffine determinism)"

        # Labels should be identical
        assert torch.allclose(datapoint1['labels']['change_map'], datapoint2['labels']['change_map']), \
            f"change_map should be identical for idx {idx}"


def test_ppsl_dataset_different_seeds(whu_bd_source_dataset):
    """Test that different seeds produce different transform results."""
    # Create two PPSLDataset instances with different seeds
    dataset1 = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=5,
        base_seed=123,
        use_cache=False
    )

    dataset2 = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=5,
        base_seed=456,
        use_cache=False
    )

    # Test that same indices produce different results due to different transform seeds
    idx = 0
    datapoint1 = dataset1[idx]
    datapoint2 = dataset2[idx]

    # Images should be different due to different transform seeds
    assert not torch.allclose(datapoint1['inputs']['img_1'], datapoint2['inputs']['img_1']), \
        "img_1 should be different with different seeds (ColorJitter)"

    assert not torch.allclose(datapoint1['inputs']['img_2'], datapoint2['inputs']['img_2']), \
        "img_2 should be different with different seeds (RandomAffine)"


def test_ppsl_dataset_same_idx_consistency(whu_bd_source_dataset):
    """Test that loading the same index multiple times gives consistent results."""
    dataset = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=5,
        base_seed=789,
        use_cache=False
    )

    # Load the same datapoint multiple times
    idx = min(1, len(dataset) - 1)  # Use index 1 or last available if dataset too small
    results = []
    for i in range(3):
        datapoint = dataset[idx]
        results.append(datapoint)

    # All results should be identical
    for i in range(1, len(results)):
        assert torch.allclose(results[0]['inputs']['img_1'], results[i]['inputs']['img_1']), \
            f"Multiple loads of same idx should be identical - img_1 differs on load {i}"

        assert torch.allclose(results[0]['inputs']['img_2'], results[i]['inputs']['img_2']), \
            f"Multiple loads of same idx should be identical - img_2 differs on load {i}"

        assert torch.allclose(results[0]['labels']['change_map'], results[i]['labels']['change_map']), \
            f"Multiple loads of same idx should be identical - change_map differs on load {i}"


def test_ppsl_dataset_different_indices(whu_bd_source_dataset):
    """Test that different indices produce different transform results."""
    dataset = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=min(5, len(whu_bd_source_dataset)),
        base_seed=999,
        use_cache=False
    )

    # Skip test if dataset has fewer than 2 datapoints
    if len(dataset) < 2:
        pytest.skip("Dataset too small for different indices test")

    # Load two different datapoints
    datapoint0 = dataset[0]
    datapoint1 = dataset[1]

    # Different indices should produce different transform results
    # (because seed is based on base_seed + idx)
    assert not torch.allclose(datapoint0['inputs']['img_1'], datapoint1['inputs']['img_1']), \
        "Different indices should produce different ColorJitter results"

    # Note: img_2 comparison is not reliable because it uses random.choice for idx_2 selection
    # which could coincidentally select the same source image and produce similar results


def test_ppsl_dataset_base_seed_none_handling(whu_bd_source_dataset):
    """Test that PPSLDataset handles None base_seed gracefully."""
    dataset = PPSLDataset(
        source=whu_bd_source_dataset,
        dataset_size=3,
        base_seed=None,  # Should default to 0
        use_cache=False
    )

    # Should be able to load datapoints without error
    datapoint = dataset[0]

    # Verify datapoint structure
    assert 'inputs' in datapoint
    assert 'labels' in datapoint
    assert 'img_1' in datapoint['inputs']
    assert 'img_2' in datapoint['inputs']
    assert 'change_map' in datapoint['labels']
