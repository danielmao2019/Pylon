"""Comprehensive tests for cache metadata functionality.

This module tests cache metadata operations including:
- Size tracking and reporting
- Memory usage monitoring  
- Metadata JSON file operations
- Cache version management
- Performance metrics
"""

import pytest
import tempfile
import os
import time
import torch
import psutil
from datetime import datetime
from unittest.mock import patch

from data.cache.cpu_dataset_cache import CPUDatasetCache
from data.cache.disk_dataset_cache import DiskDatasetCache
from data.cache.combined_dataset_cache import CombinedDatasetCache


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for testing."""
    return {
        'inputs': {
            'image': torch.randn(3, 224, 224),
            'features': torch.randn(1024)
        },
        'labels': {
            'mask': torch.randint(0, 2, (224, 224))
        },
        'meta_info': {
            'path': '/test/image.jpg',
            'index': 42
        }
    }


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestCPUCacheMetadata:
    """Test CPU cache metadata functionality."""
    
    def test_cache_size_tracking(self, sample_datapoint):
        """Test that cache correctly tracks its size."""
        cache = CPUDatasetCache(max_memory_percent=50.0)
        
        # Initial size should be 0
        assert cache.get_size() == 0
        assert len(cache.cache) == 0
        
        # Add items and verify size tracking
        for i in range(5):
            cache.put(i, sample_datapoint)
            assert cache.get_size() == i + 1
            assert len(cache.cache) == i + 1
        
        # Clear cache and verify size resets
        cache.clear()
        assert cache.get_size() == 0
        assert len(cache.cache) == 0
    
    def test_memory_usage_calculation(self, sample_datapoint):
        """Test memory usage calculation for cached items."""
        cache = CPUDatasetCache(max_memory_percent=50.0)
        
        # Calculate expected memory for sample datapoint
        expected_memory = (
            # image tensor: 3 * 224 * 224 * 4 bytes (float32)
            3 * 224 * 224 * 4 +
            # features tensor: 1024 * 4 bytes (float32) 
            1024 * 4 +
            # mask tensor: 224 * 224 * 8 bytes (int64 default)
            224 * 224 * 8 +
            # overhead: 1KB
            1024
        )
        
        # Add item and verify memory tracking
        cache.put(0, sample_datapoint)
        
        assert 0 in cache.memory_usage
        assert cache.memory_usage[0] == expected_memory
        assert cache.total_memory == expected_memory
        
        # Add second item
        cache.put(1, sample_datapoint) 
        assert cache.total_memory == 2 * expected_memory
        
        # Clear and verify memory resets
        cache.clear()
        assert cache.total_memory == 0
        assert len(cache.memory_usage) == 0
    
    def test_memory_limit_enforcement(self):
        """Test that cache respects memory limits."""
        # Use very small memory limit to force eviction
        cache = CPUDatasetCache(max_memory_percent=0.001)  # 0.001% of system memory
        
        # Create large tensors to exceed memory limit
        large_datapoint = {
            'inputs': {'image': torch.randn(10, 1000, 1000)},
            'labels': {'mask': torch.randint(0, 2, (1000, 1000))},
            'meta_info': {'index': 0}
        }
        
        # Add first item
        cache.put(0, large_datapoint)
        assert cache.get_size() == 1
        
        # Add second item - should trigger eviction of first
        cache.put(1, large_datapoint)
        
        # Cache should maintain size limit by evicting old items
        assert cache.get_size() <= 1
        assert 0 not in cache.cache  # First item should be evicted
    
    def test_cache_statistics_tracking(self, sample_datapoint):
        """Test cache hit/miss statistics and performance metrics."""
        cache = CPUDatasetCache(max_memory_percent=50.0)
        
        # Test cache misses
        result = cache.get(0)
        assert result is None
        
        # Add item and test cache hit
        cache.put(0, sample_datapoint)
        
        start_time = time.time()
        result = cache.get(0)
        end_time = time.time()
        
        assert result is not None
        assert end_time - start_time < 0.01  # Should be very fast (< 10ms)
        
        # Test LRU behavior tracking
        cache.put(1, sample_datapoint)
        cache.put(2, sample_datapoint)
        
        # Access item 0 to move it to end of LRU
        cache.get(0)
        lru_order = list(cache.cache.keys())
        assert lru_order[-1] == 0  # Most recently used


class TestDiskCacheMetadata:
    """Test disk cache metadata functionality."""
    
    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache creates proper directory structure."""
        version_hash = "test_hash_123"
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash=version_hash,
            enable_validation=True
        )
        
        # Verify directory structure
        assert os.path.exists(temp_cache_dir)
        version_dir = os.path.join(temp_cache_dir, version_hash)
        assert os.path.exists(version_dir)
        
        metadata_file = os.path.join(temp_cache_dir, 'cache_metadata.json')
        assert os.path.exists(metadata_file)
    
    def test_metadata_json_operations(self, temp_cache_dir, sample_datapoint):
        """Test metadata JSON file creation and updates."""
        version_hash = "test_hash_456"
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash=version_hash,
            enable_validation=False
        )
        
        # Check initial metadata
        metadata = cache.get_metadata()
        assert version_hash in metadata
        assert 'created_at' in metadata[version_hash]
        assert 'cache_dir' in metadata[version_hash]
        assert metadata[version_hash]['enable_validation'] is False
        
        # Create second cache with different version
        version_hash_2 = "test_hash_789"
        cache2 = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash=version_hash_2,
            enable_validation=True
        )
        
        # Verify both versions in metadata
        metadata = cache2.get_metadata()
        assert version_hash in metadata
        assert version_hash_2 in metadata
        assert metadata[version_hash_2]['enable_validation'] is True
    
    def test_disk_cache_size_tracking(self, temp_cache_dir, sample_datapoint):
        """Test disk cache size tracking."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="size_test",
            enable_validation=False
        )
        
        # Initial size should be 0
        assert cache.get_size() == 0
        
        # Add items and verify size tracking
        for i in range(3):
            cache.put(i, sample_datapoint)
            assert cache.get_size() == i + 1
            
            # Verify cache file exists
            cache_file = cache._get_cache_filepath(i)
            assert os.path.exists(cache_file)
        
        # Clear cache and verify
        cache.clear()
        assert cache.get_size() == 0
    
    def test_disk_space_usage(self, temp_cache_dir, sample_datapoint):
        """Test disk space usage tracking."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="space_test",
            enable_validation=True
        )
        
        # Add item and measure disk usage
        cache.put(0, sample_datapoint)
        
        cache_file = cache._get_cache_filepath(0)
        file_size = os.path.getsize(cache_file)
        
        # File should exist and have reasonable size
        assert file_size > 1000  # Should be at least 1KB
        assert file_size < 50 * 1024 * 1024  # Should be less than 50MB for test data
        
        # Test multiple files
        total_size_before = sum(
            os.path.getsize(os.path.join(cache.version_dir, f))
            for f in os.listdir(cache.version_dir)
            if f.endswith('.pt')
        )
        
        cache.put(1, sample_datapoint)
        
        total_size_after = sum(
            os.path.getsize(os.path.join(cache.version_dir, f))
            for f in os.listdir(cache.version_dir)
            if f.endswith('.pt')
        )
        
        assert total_size_after > total_size_before
    
    def test_cache_corruption_handling(self, temp_cache_dir, sample_datapoint):
        """Test handling of corrupted cache files."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="corruption_test",
            enable_validation=True
        )
        
        # Add valid item
        cache.put(0, sample_datapoint)
        
        # Corrupt the cache file
        cache_file = cache._get_cache_filepath(0)
        with open(cache_file, 'w') as f:
            f.write("corrupted data")
        
        # Try to retrieve - should return None and remove corrupted file
        result = cache.get(0)
        assert result is None
        assert not os.path.exists(cache_file)


class TestCombinedCacheMetadata:
    """Test combined cache metadata functionality."""
    
    def test_combined_cache_statistics(self, temp_cache_dir, sample_datapoint):
        """Test combined cache hit/miss statistics."""
        cache = CombinedDatasetCache(
            data_root=temp_cache_dir,
            version_hash="combined_test",
            use_cpu_cache=True,
            use_disk_cache=True,
            max_cpu_memory_percent=50.0
        )
        
        # Test cache miss (neither CPU nor disk)
        result = cache.get(0)
        assert result is None
        
        # Add to cache
        cache.put(0, sample_datapoint)
        
        # Should hit CPU cache first
        start_time = time.time()
        result = cache.get(0, device='cpu')
        cpu_time = time.time() - start_time
        
        assert result is not None
        assert cpu_time < 0.01  # CPU cache should be very fast
        
        # Clear CPU cache to test disk cache
        cache.cpu_cache.clear()
        
        start_time = time.time()
        result = cache.get(0, device='cpu')
        disk_time = time.time() - start_time
        
        assert result is not None
        assert disk_time > cpu_time  # Disk should be slower than CPU cache
    
    def test_cache_memory_efficiency(self, temp_cache_dir):
        """Test cache memory efficiency with large datasets."""
        cache = CombinedDatasetCache(
            data_root=temp_cache_dir,
            version_hash="efficiency_test",
            use_cpu_cache=True,
            use_disk_cache=True,
            max_cpu_memory_percent=1.0  # Very small CPU cache
        )
        
        # Add many items to test memory management
        large_datapoints = []
        for i in range(10):
            datapoint = {
                'inputs': {'image': torch.randn(3, 512, 512)},
                'labels': {'mask': torch.randint(0, 2, (512, 512))},
                'meta_info': {'index': i}
            }
            large_datapoints.append(datapoint)
            cache.put(i, datapoint)
        
        # CPU cache should have limited items due to memory constraint
        cpu_size = cache.cpu_cache.get_size()
        assert cpu_size < 10  # Should have evicted some items
        
        # Disk cache should have all items
        disk_size = cache.disk_cache.get_size()
        assert disk_size == 10
        
        # All items should be retrievable (from disk if not in CPU)
        for i in range(10):
            result = cache.get(i)
            assert result is not None


class TestCachePerformanceMetrics:
    """Test cache performance monitoring."""
    
    def test_cpu_cache_performance_monitoring(self, sample_datapoint):
        """Test CPU cache performance metrics."""
        cache = CPUDatasetCache(max_memory_percent=50.0)
        
        # Measure put performance
        put_times = []
        for i in range(100):
            start_time = time.time()
            cache.put(i, sample_datapoint)
            put_times.append(time.time() - start_time)
        
        avg_put_time = sum(put_times) / len(put_times)
        assert avg_put_time < 0.01  # Should be very fast (< 10ms average)
        
        # Measure get performance
        get_times = []
        for i in range(100):
            start_time = time.time()
            result = cache.get(i)
            get_times.append(time.time() - start_time)
            assert result is not None
        
        avg_get_time = sum(get_times) / len(get_times)
        assert avg_get_time < 0.005  # Gets should be even faster (< 5ms average)
    
    @pytest.mark.slow
    def test_disk_cache_performance_monitoring(self, temp_cache_dir, sample_datapoint):
        """Test disk cache performance metrics."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="perf_test",
            enable_validation=False  # Disable validation for pure I/O performance
        )
        
        # Measure put performance
        put_times = []
        for i in range(20):  # Fewer iterations for disk I/O
            start_time = time.time()
            cache.put(i, sample_datapoint)
            put_times.append(time.time() - start_time)
        
        avg_put_time = sum(put_times) / len(put_times)
        assert avg_put_time < 0.1  # Should be reasonably fast (< 100ms average)
        
        # Measure get performance
        get_times = []
        for i in range(20):
            start_time = time.time()
            result = cache.get(i)
            get_times.append(time.time() - start_time)
            assert result is not None
        
        avg_get_time = sum(get_times) / len(get_times)
        assert avg_get_time < 0.1  # Disk reads should be reasonably fast
    
    def test_memory_pressure_handling(self):
        """Test cache behavior under memory pressure."""
        # Mock low memory condition
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate system with only 1GB available memory
            mock_memory.return_value.total = 1024 * 1024 * 1024  # 1GB
            mock_memory.return_value.available = 100 * 1024 * 1024  # 100MB available
            
            cache = CPUDatasetCache(max_memory_percent=50.0)  # 50% of 1GB = 512MB limit
            
            # Cache should adapt to memory constraints
            max_cache_bytes = int(0.5 * 1024 * 1024 * 1024)  # 512MB
            
            # Create large datapoint that would exceed memory
            large_datapoint = {
                'inputs': {'image': torch.randn(1000, 1000, 3)},  # ~12MB
                'labels': {'mask': torch.randint(0, 2, (1000, 1000))},  # ~8MB
                'meta_info': {'index': 0}
            }
            
            # Should be able to store some items before hitting limit
            items_stored = 0
            for i in range(100):  # Try to store many items
                cache.put(i, large_datapoint)
                items_stored += 1
                
                # Stop if cache size stabilizes (indicating eviction)
                if cache.get_size() < items_stored:
                    break
            
            # Cache should have evicted items to stay under memory limit
            assert cache.total_memory <= max_cache_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])