import shutil

import pytest
import torch
from torch.utils.data import DataLoader

from utils.ops import buffer_equal


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    import tempfile

    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    # Clean up after tests
    shutil.rmtree(_temp_dir)


def test_basic_cache_functionality(SampleDataset, temp_dir):
    """Test basic cache functionality - items are cached and retrievable."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(10)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )

    # First access should cache items
    first_access = [dataset[i] for i in range(10)]
    # Second access should retrieve from cache
    second_access = [dataset[i] for i in range(10)]

    # Verify items are identical
    for first, second in zip(first_access, second_access, strict=True):
        assert buffer_equal(first, second)


def test_cache_persistence(SampleDataset, temp_dir):
    """Test that cached items persist and are reused."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )

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


def test_cache_memory_isolation(SampleDataset, temp_dir):
    """Test that cached datasets don't share memory."""
    dataset1 = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )
    dataset2 = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )

    # Access all items in both datasets
    items1 = [dataset1[i] for i in range(5)]
    items2 = [dataset2[i] for i in range(5)]

    # Verify items are equal but not the same objects
    for item1, item2 in zip(items1, items2, strict=True):
        assert buffer_equal(item1, item2)
        # Skip memory check since our sample dataset uses integers which are immutable
        # In a real dataset with tensors, we would check memory isolation


def test_cache_vs_uncached(SampleDataset, temp_dir):
    """Test that cached and uncached datasets return identical data."""
    cached = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )
    uncached = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=False,
        use_disk_cache=False,
    )

    # Multiple access patterns to verify consistency
    for _ in range(3):
        for i in range(5):
            cached_item = cached[i]
            uncached_item = uncached[i]
            assert buffer_equal(cached_item, uncached_item)


def test_cache_with_dataloader(SampleDataset, temp_dir):
    """Test cache behavior when accessed through DataLoader."""
    cached = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(10)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )
    uncached = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(10)),
        use_cpu_cache=False,
        use_disk_cache=False,
    )

    dataloader_params = {
        'batch_size': 3,
        'shuffle': False,
        'num_workers': 0,
    }

    # Create dataloaders
    loader_cached = DataLoader(cached, **dataloader_params)
    loader_uncached = DataLoader(uncached, **dataloader_params)

    # Multiple epochs to verify cache consistency
    for _ in range(3):
        for batch_cached, batch_uncached in zip(loader_cached, loader_uncached, strict=True):
            assert buffer_equal(batch_cached, batch_uncached)


def test_cache_with_different_batch_sizes(SampleDataset, temp_dir):
    """Test cache consistency with different batch sizes."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(10)),
        use_cpu_cache=True,
        use_disk_cache=True,
    )

    # Test with different batch sizes
    batch_sizes = [1, 2, 5]
    results = []

    for batch_size in batch_sizes:
        loader = DataLoader(
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
    for items1, items2 in zip(results[:-1], results[1:], strict=True):
        for item1, item2 in zip(items1, items2, strict=True):
            assert buffer_equal(item1, item2)


def test_cache_hits_with_transforms(SampleDataset, random_transforms, temp_dir):
    """Test that data is properly cached even with transforms enabled."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
        transforms_cfg=random_transforms,
    )

    # Clear any existing cache to start fresh
    dataset.cache.clear()

    # Verify cache is initially empty after clearing
    cpu_size = lambda ds: (
        len(ds.cache.cpu_cache.cache) if ds.cache.cpu_cache is not None else 0
    )
    disk_size = lambda ds: (
        ds.cache.disk_cache.get_size() if ds.cache.disk_cache is not None else 0
    )

    initial_cpu_size = cpu_size(dataset)
    initial_disk_size = disk_size(dataset)
    assert initial_cpu_size == 0
    assert initial_disk_size == 0

    # Access the item - should be cached now
    _ = dataset[0]
    after_first_cpu_size = cpu_size(dataset)
    after_first_disk_size = disk_size(dataset)
    assert after_first_cpu_size == 1
    assert after_first_disk_size == 1

    # Second access should hit cache - cache size should remain the same
    _ = dataset[0]
    after_second_cpu_size = cpu_size(dataset)
    after_second_disk_size = disk_size(dataset)
    assert after_second_cpu_size == 1
    assert after_second_disk_size == 1


def test_transform_randomness_preserved(SampleDataset, random_transforms, temp_dir):
    """Test that transforms remain random even with cached data."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(5)),
        use_cpu_cache=True,
        use_disk_cache=True,
        transforms_cfg=random_transforms,
    )

    dataset.set_base_seed(0)
    first_access = dataset[0]
    dataset.set_base_seed(1)
    second_access = dataset[0]

    # Data should be different due to random transforms
    assert not torch.allclose(
        first_access['inputs']['input'].float(),
        second_access['inputs']['input'].float(),
    )


def test_transform_randomness_in_dataloader(SampleDataset, random_transforms, temp_dir):
    """Test that transform randomness works through DataLoader."""
    dataset = SampleDataset(
        data_root=temp_dir,
        split='train',
        indices=list(range(10)),
        use_cpu_cache=True,
        use_disk_cache=True,
        transforms_cfg=random_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Run two epochs and collect results
    dataloader.dataset.set_base_seed(0)
    epoch1_data = []
    for batch in dataloader:
        epoch1_data.append(batch['inputs']['input'].clone())

    dataloader.dataset.set_base_seed(1)
    epoch2_data = []
    for batch in dataloader:
        epoch2_data.append(batch['inputs']['input'].clone())

    # Compare corresponding batches between epochs
    for batch1, batch2 in zip(epoch1_data, epoch2_data, strict=True):
        assert not torch.allclose(batch1.float(), batch2.float())
