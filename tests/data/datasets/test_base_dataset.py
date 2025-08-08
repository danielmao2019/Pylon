import pytest
import torch


def test_base_dataset_split_none(SampleDatasetWithoutPredefinedSplits) -> None:
    """Test BaseDataset with split=None (load everything)."""
    dataset = SampleDatasetWithoutPredefinedSplits(
        split=None,
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    
    # Verify dataset attributes
    assert dataset.split is None
    assert not hasattr(dataset, 'split_percentages')
    assert dataset.indices is None
    assert len(dataset) == 100  # Should load all data
    
    # Verify data integrity - should contain all labels 0-99
    all_labels = set()
    for i in range(len(dataset)):
        datapoint = dataset[i]
        label = datapoint['labels']['label']
        all_labels.add(label)
        
        # Verify tensor properties
        input_tensor = datapoint['inputs']['input']
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (3, 32, 32)
        assert input_tensor.device == torch.device('cpu')
        
        # Verify deterministic generation
        torch.manual_seed(label)
        expected_tensor = torch.randn(3, 32, 32)
        assert torch.allclose(input_tensor, expected_tensor)
    
    assert all_labels == set(range(100)), f"Missing labels: {set(range(100)) - all_labels}"


def test_base_dataset_split_none_with_indices(SampleDatasetWithoutPredefinedSplits) -> None:
    """Test BaseDataset with split=None and custom indices."""
    indices = [10, 20, 30, 40, 50]
    dataset = SampleDatasetWithoutPredefinedSplits(
        split=None,
        indices=indices,
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    
    # Verify dataset attributes
    assert dataset.split is None
    assert dataset.indices == indices
    assert len(dataset) == len(indices)
    
    # Verify data matches indices
    expected_labels = [10, 20, 30, 40, 50]
    for i, expected_label in enumerate(expected_labels):
        datapoint = dataset[i]
        assert datapoint['labels']['label'] == expected_label


def test_base_dataset_predefined_splits(SampleDataset) -> None:
    """Test BaseDataset with predefined splits."""
    # Test train split
    train_dataset = SampleDataset(
        split='train',
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    assert train_dataset.split == 'train'
    assert len(train_dataset) == 80
    
    # Verify train data range (0-79)
    train_labels = {train_dataset[i]['labels']['label'] for i in range(len(train_dataset))}
    assert train_labels == set(range(80))
    
    # Test val split
    val_dataset = SampleDataset(
        split='val',
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    assert val_dataset.split == 'val'
    assert len(val_dataset) == 10
    
    # Verify val data range (80-89)
    val_labels = {val_dataset[i]['labels']['label'] for i in range(len(val_dataset))}
    assert val_labels == set(range(80, 90))
    
    # Test test split
    test_dataset = SampleDataset(
        split='test',
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    assert test_dataset.split == 'test'
    assert len(test_dataset) == 10
    
    # Verify test data range (90-99)
    test_labels = {test_dataset[i]['labels']['label'] for i in range(len(test_dataset))}
    assert test_labels == set(range(90, 100))
    
    # Test empty split
    weird_dataset = SampleDataset(
        split='weird',
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    assert weird_dataset.split == 'weird'
    assert len(weird_dataset) == 0


def test_base_dataset_predefined_splits_with_indices(SampleDataset) -> None:
    """Test BaseDataset predefined splits with custom indices."""
    indices = [1, 3, 5]  # Select specific indices from train split
    train_dataset = SampleDataset(
        split='train',
        indices=indices,
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False
    )
    
    assert train_dataset.split == 'train'
    assert train_dataset.indices == indices
    assert len(train_dataset) == len(indices)
    
    # Verify data matches indices (should be labels 1, 3, 5 from train range)
    expected_labels = [1, 3, 5]
    for i, expected_label in enumerate(expected_labels):
        datapoint = train_dataset[i]
        assert datapoint['labels']['label'] == expected_label


def test_base_dataset_split_percentages(SampleDatasetWithoutPredefinedSplits) -> None:
    """Test BaseDataset with split_percentages."""
    # Test with specific percentages
    train_dataset = SampleDatasetWithoutPredefinedSplits(
        split='train',
        split_percentages=(0.8, 0.1, 0.1, 0.0),
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False,
        base_seed=42  # Fixed seed for reproducibility
    )
    
    assert train_dataset.split == 'train'
    assert train_dataset.split_percentages == (0.8, 0.1, 0.1, 0.0)
    assert len(train_dataset) == 80  # 80% of 100
    
    # Test val split with same percentages
    val_dataset = SampleDatasetWithoutPredefinedSplits(
        split='val',
        split_percentages=(0.8, 0.1, 0.1, 0.0),
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False,
        base_seed=42  # Same seed should produce same splits
    )
    
    assert val_dataset.split == 'val'
    assert len(val_dataset) == 10  # 10% of 100
    
    # Verify splits are disjoint (no overlap in labels)
    train_labels = {train_dataset[i]['labels']['label'] for i in range(len(train_dataset))}
    val_labels = {val_dataset[i]['labels']['label'] for i in range(len(val_dataset))}
    assert train_labels.isdisjoint(val_labels), "Train and val splits should not overlap"
    
    # Test different seed produces different splits
    train_dataset_diff_seed = SampleDatasetWithoutPredefinedSplits(
        split='train',
        split_percentages=(0.8, 0.1, 0.1, 0.0),
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False,
        base_seed=123  # Different seed
    )
    
    train_labels_diff_seed = {train_dataset_diff_seed[i]['labels']['label'] for i in range(len(train_dataset_diff_seed))}
    assert train_labels != train_labels_diff_seed, "Different seeds should produce different splits"


def test_base_dataset_split_percentages_with_indices(SampleDatasetWithoutPredefinedSplits) -> None:
    """Test BaseDataset with split_percentages and custom indices."""
    indices = [0, 2, 4, 6]  # Select specific indices after split
    train_dataset = SampleDatasetWithoutPredefinedSplits(
        split='train',
        split_percentages=(0.8, 0.1, 0.1, 0.0),
        indices=indices,
        device=torch.device('cpu'),
        use_cpu_cache=False,
        use_disk_cache=False,
        base_seed=42
    )
    
    assert train_dataset.split == 'train'
    assert train_dataset.indices == indices
    assert len(train_dataset) == len(indices)


def test_base_dataset_invalid_configurations(SampleDataset, SampleDatasetWithoutPredefinedSplits) -> None:
    """Test invalid BaseDataset configurations."""
    
    # Test split=None with split_percentages should fail
    with pytest.raises(AssertionError, match="Cannot use split_percentages when split=None"):
        SampleDatasetWithoutPredefinedSplits(
            split=None,
            split_percentages=(0.8, 0.1, 0.1, 0.0),
            device=torch.device('cpu'),
            use_cpu_cache=False,
            use_disk_cache=False
        )
    
    # Test invalid split name should fail
    with pytest.raises(AssertionError, match="not in"):
        SampleDataset(
            split='invalid',
            device=torch.device('cpu'),
            use_cpu_cache=False,
            use_disk_cache=False
        )
    
    # Test split_percentages that don't sum to 1.0 should fail
    with pytest.raises(AssertionError, match="Percentages must sum to 1.0"):
        SampleDatasetWithoutPredefinedSplits(
            split='train',
            split_percentages=(0.5, 0.3, 0.1, 0.05),  # Sums to 0.95
            device=torch.device('cpu'),
            use_cpu_cache=False,
            use_disk_cache=False
        )
    
    # Test wrong number of percentages should fail
    with pytest.raises(AssertionError):
        SampleDatasetWithoutPredefinedSplits(
            split='train',
            split_percentages=(0.8, 0.2),  # Only 2 values for 4 splits
            device=torch.device('cpu'),
            use_cpu_cache=False,
            use_disk_cache=False
        )
