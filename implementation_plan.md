# Disk-Based Dataset Cache Implementation Plan

## Overview
Add a disk-based caching mechanism alongside the existing RAM cache to enable persistent storage of raw datapoints. The cache will save individual datapoints as separate files using torch.save for efficient GPU memory mapping.

## Key Design Decisions

### 1. Cache Directory Structure
```
<data_root>_cache/
├── <dataset_version_hash>/
│   ├── 0.pt
│   ├── 1.pt
│   ├── 2.pt
│   └── ...
└── cache_metadata.json
```
Cache directory is always `<data_root>_cache` (sibling to data root, no overrides allowed).

### 2. Dataset Version Hashing
To handle different dataset configurations, we'll create a deterministic hash:

```python
def generate_dataset_version_hash(dataset_instance) -> str:
    """Generate deterministic hash from dataset configuration."""
    # Extract all parameters that affect dataset content
    hash_dict = {
        'class_name': dataset_instance.__class__.__name__,
        'data_root': str(dataset_instance.data_root),
    }
    
    # Add split information
    if hasattr(dataset_instance, 'split') and dataset_instance.split is not None:
        hash_dict['split'] = dataset_instance.split
    elif hasattr(dataset_instance, 'split_percentages'):
        hash_dict['split_percentages'] = dataset_instance.split_percentages
    
    # Add dataset-specific parameters
    if hasattr(dataset_instance, 'rotation_mag'):
        hash_dict['rotation_mag'] = dataset_instance.rotation_mag
    if hasattr(dataset_instance, 'translation_mag'):
        hash_dict['translation_mag'] = dataset_instance.translation_mag
    if hasattr(dataset_instance, 'matching_radius'):
        hash_dict['matching_radius'] = dataset_instance.matching_radius
    if hasattr(dataset_instance, 'camera_count'):
        hash_dict['camera_count'] = dataset_instance.camera_count
    if hasattr(dataset_instance, 'total_dataset_size'):
        hash_dict['dataset_size'] = dataset_instance.total_dataset_size
    
    # Create deterministic string representation and hash it
    hash_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]  # Use first 16 chars
```

### 3. Implementation Components

#### A. RAMDatasetCache Class (renamed from DatasetCache)
- Rename existing `DatasetCache` to `RAMDatasetCache` for clarity
- Make checksum validation mandatory (remove `enable_validation` parameter)
- Remove all statistics tracking (hits, misses, stats methods)

#### B. DiskDatasetCache Class
```python
class DiskDatasetCache:
    """Disk-based cache for dataset items with per-datapoint files."""
    
    def __init__(
        self,
        cache_dir: str,
        version_hash: str,
    ):
        # Initialize disk cache with specific version directory
        # Checksums are mandatory
        
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        # Load from disk if exists, validate checksum
        
    def put(self, idx: int, value: Dict[str, Any]) -> None:
        # Save to disk with checksum
        
    def exists(self, idx: int) -> bool:
        # Check if cache file exists
```

#### C. CombinedDatasetCache Class
```python
class CombinedDatasetCache:
    """Unified cache interface combining RAM and disk caching."""
    
    def __init__(
        self,
        data_root: str,
        version_hash: str,
        use_ram_cache: bool = True,
        use_disk_cache: bool = True,
        max_ram_memory_percent: float = 80.0,
    ):
        # Initialize both RAM and disk caches
        # Cache directory is always <data_root>_cache
        
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        # Check RAM → Disk → None
        
    def put(self, idx: int, value: Dict[str, Any]) -> None:
        # Store in both RAM and disk caches
```

#### D. Integration with BaseDataset
Modify `BaseDataset.__init__` to:
1. Generate version hash from dataset configuration
2. Use single `CombinedDatasetCache` instance
3. Implement cache hierarchy: RAM → Disk → Load from source

### 4. Cache File Format
Each cache file will contain:
```python
{
    'inputs': {...},      # Raw input tensors
    'labels': {...},      # Raw label tensors  
    'meta_info': {...},   # Metadata
    'checksum': '...',    # Mandatory validation checksum
}
```

### 5. Configuration Options
Add new parameters to BaseDataset:
```python
use_ram_cache: bool = True   # Enable RAM caching (renamed from use_cache)
use_disk_cache: bool = True  # Enable disk caching
max_cache_memory_percent: float = 80.0  # RAM cache memory limit
```
Note: Cache directory is always `<data_root>_cache`, no override option.

### 6. Benefits
- **Persistent Storage**: Cache survives across sessions
- **GPU Direct Loading**: Use `torch.load(map_location='cuda:0')` for direct GPU transfer
- **Version Management**: Different dataset configurations don't conflict
- **Single File Per Datapoint**: Efficient for small batch sizes
- **Parallel Access**: Multiple processes can read cache files simultaneously

### 7. Implementation Order
1. Rename `DatasetCache` to `RAMDatasetCache` and remove statistics
2. Implement version hash generation utility
3. Create `DiskDatasetCache` class
4. Create `CombinedDatasetCache` class
5. Modify `BaseDataset` to use combined cache
6. Update dataset loading logic with cache hierarchy
7. Add cache management utilities (clear old versions)

### 8. Considerations
- **Thread Safety**: Use file locks for write operations
- **Disk Space**: Monitor available disk space
- **Cleanup**: Provide utilities to clear old cache versions
- **Validation**: Mandatory checksum validation for both caches
- **Compression**: Consider torch.save with compression for large datasets
- **Simplicity**: No statistics tracking to keep implementation clean
