# 3DMatch Metadata Comparison: GeoTransformer vs OverlapPredator

## Executive Summary

Both GeoTransformer and OverlapPredator repositories contain metadata files for the 3DMatch and 3DLoMatch benchmarks, but they organize and structure this data differently. GeoTransformer uses a **list-of-dictionaries** format with each entry as a separate dictionary, while OverlapPredator uses a **dictionary-of-lists** format where data is organized by field type.

## Directory Structure

### GeoTransformer
```
data/3DMatch/metadata/
├── 3DMatch.pkl
├── 3DLoMatch.pkl
├── train.pkl
├── val.pkl
├── split/
│   ├── train_3dmatch.txt
│   └── val_3dmatch.txt
└── benchmarks/
    ├── 3DMatch/
    └── 3DLoMatch/
```

### OverlapPredator
```
configs/indoor/
├── 3DMatch.pkl
├── 3DLoMatch.pkl
├── train_info.pkl
├── val_info.pkl
├── train_3dmatch.txt
└── val_3dmatch.txt
```

## File-by-File Comparison

### 1. Test Benchmark Files (3DMatch.pkl & 3DLoMatch.pkl)

#### GeoTransformer Format
- **Structure**: List of dictionaries
- **3DMatch.pkl**: 1,623 entries
- **3DLoMatch.pkl**: 1,781 entries
- **Entry format**:
  ```python
  {
    'overlap': float64,          # Overlap ratio
    'pcd0': str,                 # Path to first point cloud
    'pcd1': str,                 # Path to second point cloud
    'rotation': ndarray(3,3),    # Rotation matrix
    'translation': ndarray(3,),   # Translation vector
    'scene_name': str,           # Scene identifier
    'frag_id0': int,             # Fragment ID for pcd0
    'frag_id1': int              # Fragment ID for pcd1
  }
  ```

#### OverlapPredator Format
- **Structure**: Dictionary of lists
- **3DMatch.pkl**: 1,623 entries
- **3DLoMatch.pkl**: 1,781 entries
- **Dictionary format**:
  ```python
  {
    'rot': [ndarray(3,3), ...],      # List of rotation matrices
    'trans': [ndarray(3,1), ...],    # List of translation vectors
    'src': [str, ...],               # List of source point cloud paths
    'tgt': [str, ...],               # List of target point cloud paths
    'overlap': [float64, ...]        # List of overlap ratios
  }
  ```

**Key Differences**:
1. **Data organization**: Row-oriented (GeoTransformer) vs Column-oriented (OverlapPredator)
2. **Translation shape**: (3,) in GeoTransformer vs (3,1) in OverlapPredator
3. **Path naming**: `pcd0/pcd1` vs `src/tgt`
4. **Path order**: Note that OverlapPredator swaps the order - `src` corresponds to `pcd1` and `tgt` to `pcd0`
5. **Additional metadata**: GeoTransformer includes `scene_name` and fragment IDs

### 2. Training Files

#### GeoTransformer (train.pkl)
- **Structure**: List of 20,642 dictionaries
- **Format**: Same as test files (list of individual entries)
- **Fields**: Same 8 fields as test data

#### OverlapPredator (train_info.pkl)
- **Structure**: Dictionary with arrays
- **Total entries**: 20,642
- **Format**:
  ```python
  {
    'src': [str, ...],                    # List of source paths
    'tgt': [str, ...],                    # List of target paths
    'rot': ndarray(20642, 3, 3),         # Stacked rotation matrices
    'trans': ndarray(20642, 3, 1),       # Stacked translation vectors
    'overlap': ndarray(20642,)           # Array of overlap values
  }
  ```

**Key Differences**:
1. **Storage efficiency**: OverlapPredator uses numpy arrays for transformations (more memory-efficient)
2. **Access pattern**: GeoTransformer optimized for random access, OverlapPredator for batch processing

### 3. Validation Files

#### GeoTransformer (val.pkl)
- **Structure**: List of 1,331 dictionaries
- **Format**: Same as train.pkl

#### OverlapPredator (val_info.pkl)
- **Structure**: Dictionary with arrays
- **Total entries**: 1,331
- **Format**: Same structure as train_info.pkl but smaller

### 4. Split Text Files

Both repositories contain identical text files listing scene names:
- **train_3dmatch.txt**: 54 training scenes (identical content)
- **val_3dmatch.txt**: 6 validation scenes (identical content)

The only difference is the location:
- GeoTransformer: `data/3DMatch/metadata/split/`
- OverlapPredator: `configs/indoor/`

## Unique Files

### GeoTransformer Only
1. **benchmarks/** directory: Contains scene-specific subdirectories for both 3DMatch and 3DLoMatch benchmarks
   - Organized by scene name
   - Likely contains preprocessed or cached data

### OverlapPredator Only
- No unique metadata files (all have counterparts in GeoTransformer)

## Data Consistency

### Matching Data Points
- Both repositories have **exactly the same number of entries**:
  - 3DMatch test: 1,623 pairs
  - 3DLoMatch test: 1,781 pairs
  - Training: 20,642 pairs
  - Validation: 1,331 pairs

### Path References
- Both use the same path structure: `{split}/{scene_name}/cloud_bin_{id}.pth`
- Example: `test/7-scenes-redkitchen/cloud_bin_0.pth`

## Summary of Key Differences

| Aspect | GeoTransformer | OverlapPredator |
|--------|---------------|-----------------|
| **Location** | `data/3DMatch/metadata/` | `configs/indoor/` |
| **Data Structure** | List of dictionaries | Dictionary of lists/arrays |
| **Translation Shape** | (3,) | (3,1) |
| **Naming Convention** | pcd0/pcd1 | src/tgt (swapped order) |
| **Storage Format** | Individual entries | Batched numpy arrays (train/val) |
| **Additional Metadata** | scene_name, frag_id0, frag_id1 | None |
| **Benchmarks Folder** | Yes | No |
| **Memory Efficiency** | Lower (redundant Python objects) | Higher (numpy arrays) |
| **Access Pattern** | Optimized for random access | Optimized for batch processing |

## Recommendations for Usage

1. **GeoTransformer format** is better for:
   - Random access to individual pairs
   - Preserving complete metadata
   - Debugging and inspection

2. **OverlapPredator format** is better for:
   - Batch processing
   - Memory efficiency
   - Direct numpy operations

Both formats contain the same core information and can be converted between each other with appropriate transformations.