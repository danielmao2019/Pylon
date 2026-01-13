def test_cache_memory_management(three_item_cache, make_datapoint, cache_key_factory):
    """Test memory management and eviction."""
    cache = three_item_cache
    tensors = []  # Keep references to prevent garbage collection

    # Create and store initial items
    for i in range(3):
        datapoint = make_datapoint(i)
        tensors.append(datapoint['inputs']['image'])
        cache.put(datapoint, cache_filepath=cache_key_factory(i))

    # Verify initial state
    assert len(cache.cache) == 3, "Cache should have exactly 3 items"

    # Add one more item to trigger eviction
    extra_datapoint = make_datapoint(3)
    tensors.append(extra_datapoint['inputs']['image'])
    cache.put(extra_datapoint, cache_filepath=cache_key_factory(3))

    # Verify eviction
    assert len(cache.cache) == 3, "Cache should maintain 3 items"
    assert cache_key_factory(0) not in cache.cache, "First item should have been evicted"
    assert cache_key_factory(3) in cache.cache, "New item should be present"

    # Cleanup
    del tensors
    import gc
    gc.collect()
