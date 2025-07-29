"""Performance and scalability tests for eval_viewer components."""

import pytest
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor
from runners.eval_viewer.backend.visualization import create_score_map_grid, create_overlaid_score_map


def test_large_dataset_handling():
    """Test handling of larger datasets for performance validation."""
    # Create a large synthetic dataset
    num_datapoints = 10000
    scores = list(np.random.random(num_datapoints))
    
    # Test that large datasets are handled efficiently
    score_map = create_score_map_grid(scores)
    
    # Verify correct grid size (ceil(sqrt(10000)) = 100)
    assert score_map.shape == (100, 100)
    
    # Verify first N values match input
    for i in range(num_datapoints):
        assert score_map.flat[i] == scores[i]
    
    # Verify remaining positions are NaN
    remaining_positions = 100 * 100 - num_datapoints
    assert np.sum(np.isnan(score_map.flat[num_datapoints:])) == remaining_positions


def test_concurrent_processing_compatibility():
    """Test that functions work correctly when called concurrently."""
    # Create multiple score arrays
    score_arrays = [list(np.random.random(50)) for _ in range(10)]
    
    def process_scores(scores):
        return create_score_map_grid(scores)
    
    # Process arrays concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_scores, score_arrays))
    
    # Verify all results are valid
    assert len(results) == 10
    for result in results:
        assert isinstance(result, np.ndarray)
        assert result.shape == (8, 8)  # ceil(sqrt(50)) = 8


def test_memory_cleanup_with_large_data():
    """Test that memory is properly managed with large data structures."""
    # Get initial memory usage
    initial_objects = len(gc.get_objects())
    
    # Create large data structures
    large_score_maps = []
    for i in range(5):
        large_map = np.random.random((1000, 1000))
        large_score_maps.append(large_map)
    
    # Process large data
    result = create_overlaid_score_map(large_score_maps, percentile=25)
    
    # Verify result is correct size
    assert result.shape == (1000, 1000)
    
    # Clean up
    del large_score_maps
    del result
    gc.collect()
    
    # Verify memory is released (allow some variance)
    final_objects = len(gc.get_objects())
    assert final_objects < initial_objects + 100  # Allow some overhead