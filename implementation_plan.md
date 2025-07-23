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

### 2. Dataset Version Hashing
To handle different dataset configurations, we'll create a deterministic hash from:
- Dataset class name
- Split (train/val/test) or split_percentages
- Critical parameters that affect dataset content:
  - For synthetic datasets: rotation_mag, translation_mag, matching_radius, dataset_size
  - For LiDAR camera pose: camera_count
  - Any other dataset-specific parameters
- Data root path (to handle same dataset class on different data)
- Dataset version (if available)

### 3. Implementation Components

#### A. DiskDatasetCache Class
```python
class DiskDatasetCache:
    """Disk-based cache for dataset items with per-datapoint files."""
    
    def __init__(
        self,
        cache_dir: str,
        version_hash: str,
        enable_validation: bool = True,
    ):
        # Initialize disk cache with specific version directory
        
    def get(self, idx: int) -> Optional[Dict[str, Any]]:
        # Load from disk if exists
        
    def put(self, idx: int, value: Dict[str, Any]) -> None:
        # Save to disk as individual file
        
    def exists(self, idx: int) -> bool:
        # Check if cache file exists
```

#### B. Dataset Version Hash Generation
```python
def generate_dataset_version_hash(
    dataset_class_name: str,
    split: Optional[Union[str, Tuple[float, ...]]],
    dataset_params: Dict[str, Any],
    data_root: str,
) -> str:
    """Generate deterministic hash for dataset version."""
    # Create hash from all parameters that affect dataset content
```

#### C. Integration with BaseDataset
Modify `BaseDataset.__init__` to:
1. Generate version hash from dataset configuration
2. Initialize both RAM and disk caches
3. Use a cache hierarchy: RAM → Disk → Load from source

### 4. Cache File Format
Each cache file will contain:
```python
{
    'inputs': {...},      # Raw input tensors
    'labels': {...},      # Raw label tensors  
    'meta_info': {...},   # Metadata
    'checksum': '...',    # Optional validation checksum
}
```

### 5. Configuration Options
Add new parameters to BaseDataset:
```python
use_disk_cache: bool = True  # Enable disk caching
disk_cache_dir: Optional[str] = None  # Override default cache directory
cache_version_params: Optional[Dict[str, Any]] = None  # Additional version params
```

### 6. Benefits
- **Persistent Storage**: Cache survives across sessions
- **GPU Direct Loading**: Use `torch.load(map_location='cuda:0')` for direct GPU transfer
- **Version Management**: Different dataset configurations don't conflict
- **Single File Per Datapoint**: Efficient for small batch sizes
- **Parallel Access**: Multiple processes can read cache files simultaneously

### 7. Implementation Order
1. Create `DiskDatasetCache` class
2. Implement version hash generation utility
3. Modify `BaseDataset` to support disk cache
4. Add configuration options
5. Update dataset loading logic with cache hierarchy
6. Add cache management utilities (clear, stats, etc.)

### 8. Considerations
- **Thread Safety**: Use file locks for write operations
- **Disk Space**: Monitor available disk space
- **Cleanup**: Provide utilities to clear old cache versions
- **Validation**: Optional checksum validation like RAM cache
- **Compression**: Consider torch.save with compression for large datasets