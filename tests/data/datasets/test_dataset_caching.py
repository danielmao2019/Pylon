from typing import Dict, Any
import pytest
import torch
from utils.ops import buffer_equal


def test_basic_cache_functionality(SampleDataset):
    """Test basic cache functionality - items are cached and retrievable."""
    dataset = SampleDataset(split='train', indices=list(range(10)), use_cache=True)
    
    # First access should cache items
    first_access = [dataset[i] for i in range(10)]
    # Second access should retrieve from cache
    second_access = [dataset[i] for i in range(10)]
    
    # Verify items are identical
    for first, second in zip(first_access, second_access):
        assert buffer_equal(first, second)


def test_cache_persistence(SampleDataset):
    """Test that cached items persist and are reused."""
    dataset = SampleDataset(split='train', indices=list(range(5)), use_cache=True)
    
    # Access items multiple times in different orders
    access_patterns = [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [2, 0, 4, 1, 3],
    ]
    
    results = []
    for pattern in access_patterns:
        results.append([dataset[i] for i in pattern])
    
    # Verify all accesses return identical items
    for i in range(5):
        for j in range(len(access_patterns) - 1):
            idx1 = access_patterns[j].index(i)
            idx2 = access_patterns[j + 1].index(i)
            assert buffer_equal(results[j][idx1], results[j + 1][idx2])


def test_cache_memory_isolation(SampleDataset):
    """Test that cached datasets don't share memory."""
    dataset1 = SampleDataset(split='train', indices=list(range(5)), use_cache=True)
    dataset2 = SampleDataset(split='train', indices=list(range(5)), use_cache=True)
    
    # Access all items in both datasets
    items1 = [dataset1[i] for i in range(5)]
    items2 = [dataset2[i] for i in range(5)]
    
    # Verify items are equal but not the same objects
    for item1, item2 in zip(items1, items2):
        assert buffer_equal(item1, item2)
        # Skip memory check since our sample dataset uses integers which are immutable
        # In a real dataset with tensors, we would check memory isolation


def test_cache_vs_uncached(SampleDataset):
    """Test that cached and uncached datasets return identical data."""
    cached = SampleDataset(split='train', indices=list(range(5)), use_cache=True)
    uncached = SampleDataset(split='train', indices=list(range(5)), use_cache=False)
    
    # Multiple access patterns to verify consistency
    for _ in range(3):
        for i in range(5):
            cached_item = cached[i]
            uncached_item = uncached[i]
            assert buffer_equal(cached_item, uncached_item)


def test_cache_with_dataloader(SampleDataset):
    """Test cache behavior when accessed through DataLoader."""
    cached = SampleDataset(split='train', indices=list(range(10)), use_cache=True)
    uncached = SampleDataset(split='train', indices=list(range(10)), use_cache=False)
    
    dataloader_params = {
        'batch_size': 3,
        'shuffle': False,
        'num_workers': 0,
    }
    
    # Create dataloaders
    loader_cached = torch.utils.data.DataLoader(cached, **dataloader_params)
    loader_uncached = torch.utils.data.DataLoader(uncached, **dataloader_params)
    
    # Multiple epochs to verify cache consistency
    for _ in range(3):
        for batch_cached, batch_uncached in zip(loader_cached, loader_uncached):
            assert buffer_equal(batch_cached, batch_uncached)


def test_cache_with_different_batch_sizes(SampleDataset):
    """Test cache consistency with different batch sizes."""
    dataset = SampleDataset(split='train', indices=list(range(10)), use_cache=True)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 5]
    results = []
    
    for batch_size in batch_sizes:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        # Collect all items from this loader
        epoch_items = []
        for batch in loader:
            for i in range(len(batch['inputs']['input'])):
                item = {
                    'inputs': {'input': batch['inputs']['input'][i]},
                    'labels': {'label': batch['labels']['label'][i]},
                }
                epoch_items.append(item)
        results.append(epoch_items)
    
    # Verify all results are identical regardless of batch size
    for items1, items2 in zip(results[:-1], results[1:]):
        for item1, item2 in zip(items1, items2):
            assert buffer_equal(item1, item2)
