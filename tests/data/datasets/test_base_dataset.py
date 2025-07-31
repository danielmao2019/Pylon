import pytest
import torch


@pytest.fixture
def dataset_with_params(request, SampleDataset):
    """Fixture for creating a SampleDataset instance with parameters."""
    indices, expected_indices = request.param
    dataset = SampleDataset(split=None, indices=indices, device=torch.device('cpu'), use_cpu_cache=False, use_disk_cache=False)
    return dataset, expected_indices, indices


@pytest.mark.parametrize("dataset_with_params", [
    (
        None,
        {'train': list(range(100)), 'val': list(range(100)), 'test': list(range(100)), 'weird': list(range(100))},
    ),
    (
        {'train': [2, 4, 6, 8]},
        {'train': [2, 4, 6, 8], 'val': list(range(100)), 'test': list(range(100)), 'weird': list(range(100))},
    ),
    (
        {'train': [2], 'val': [2, 4], 'test': [2, 4, 6], 'weird': [2, 4, 6, 8]},
        {'train': [2], 'val': [2, 4], 'test': [2, 4, 6], 'weird': [2, 4, 6, 8]},
    ),
], indirect=True)
def test_base_dataset_None(
    dataset_with_params,
) -> None:
    # Create dataset with CPU device for testing
    dataset, expected_indices, indices = dataset_with_params
    assert dataset.split is None and not hasattr(dataset, 'split_percentage')
    assert not hasattr(dataset, 'indices') and hasattr(dataset, 'split_indices')
    assert hasattr(dataset, 'split_subsets')
    assert list(dataset.split_subsets.keys()) == ['train', 'val', 'test', 'weird']
    for split in ['train', 'val', 'test', 'weird']:
        split_subset = dataset.split_subsets[split]
        assert split_subset.split == split, f"{split_subset.split=}, {split=}"
        assert split_subset.indices == (indices.get(split, None) if indices else None)
        assert not hasattr(split_subset, 'split_indices')
        assert not hasattr(split_subset, 'split_subsets')

        # For each index, verify the label matches the index
        for idx, datapoint in enumerate(split_subset):
            expected_idx = expected_indices[split][idx]
            assert datapoint['labels']['label'] == expected_idx

            # Verify input tensor properties
            input_tensor = datapoint['inputs']['input']
            assert isinstance(input_tensor, torch.Tensor)
            assert input_tensor.shape == (3, 32, 32)
            assert input_tensor.device == torch.device('cpu')
            # Verify that the tensor is deterministic for each index by checking it against a fresh one
            torch.manual_seed(expected_idx)
            expected_tensor = torch.randn(3, 32, 32)
            assert torch.allclose(input_tensor, expected_tensor)


@pytest.fixture
def dataset_with_tuple_params(request, SampleDataset):
    """Fixture for creating a SampleDataset instance with tuple split parameters."""
    split, expected = request.param
    dataset = SampleDataset(split=split, indices=None, device=torch.device('cpu'), use_cpu_cache=False, use_disk_cache=False)
    return dataset, expected


@pytest.mark.parametrize("dataset_with_tuple_params", [
    (
        (0.8, 0.1, 0.1, 0.0),
        {'train': 80, 'val': 10, 'test': 10, 'weird': 0},
    ),
    (
        (0.7, 0.1, 0.1, 0.1),
        {'train': 70, 'val': 10, 'test': 10, 'weird': 10},
    ),
], indirect=True)
def test_base_dataset_tuple(
    dataset_with_tuple_params,
) -> None:
    # Create dataset with CPU device for testing
    dataset, expected = dataset_with_tuple_params
    assert not hasattr(dataset, 'split') and type(dataset.split_percentages) == tuple
    assert not hasattr(dataset, 'indices') and not hasattr(dataset, 'split_indices')
    assert hasattr(dataset, 'split_subsets')
    assert list(dataset.split_subsets.keys()) == ['train', 'val', 'test', 'weird']
    for split in ['train', 'val', 'test', 'weird']:
        split_subset = dataset.split_subsets[split]
        assert split_subset.split == split, f"{split_subset.split=}, {split=}"
        assert not hasattr(split_subset, 'split_percentages')
        assert not hasattr(split_subset, 'indices') and not hasattr(split_subset, 'split_indices')
        assert not hasattr(split_subset, 'split_subsets')
        assert len(split_subset) == expected[split], \
            f"{split=}, {len(split_subset)=}, {expected[split]=}"

    # Verify that splits are disjoint by checking labels
    all_labels = {split: set(datapoint['labels']['label'] for datapoint in dataset.split_subsets[split])
                 for split in ['train', 'val', 'test', 'weird']}
    for split1 in ['train', 'val', 'test', 'weird']:
        for split2 in ['train', 'val', 'test', 'weird']:
            if split1 != split2:
                assert all_labels[split1].isdisjoint(all_labels[split2]), \
                    f"Labels in {split1} and {split2} overlap"
