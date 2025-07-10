# 3DMatch Dataset Implementation Plan

## Overview
Implementing `data/datasets/pcr_datasets/threedmatch_dataset.py` following Pylon's dataset design patterns while incorporating functionality from reference implementations.

## 1. Dataset Structure Analysis

### 1.1 File System Structure
- **Metadata location**: `/home/daniel/repos/pcr-repos/GeoTransformer/data/3DMatch/metadata/`
  - `train.pkl`, `val.pkl` - main metadata files
  - `3DMatch.pkl`, `3DLoMatch.pkl` - benchmark metadata
  - Split files: `train_3dmatch.txt`, `val_3dmatch.txt`
  
- **Data location**: Not currently accessible, but based on metadata:
  - Point cloud files: `.pth` format (PyTorch tensors)
  - Path structure: `{split}/{scene_name}/cloud_bin_{id}.pth`
  - Example: `train/rgbd-scenes-v2-scene_01/cloud_bin_2.pth`

### 1.2 Metadata Format
Both GeoTransformer and OverlapPredator use the same metadata structure:
```python
{
    'overlap': float,  # Overlap ratio between point cloud pairs
    'pcd0': str,       # Path to source point cloud (GeoTransformer)
    'pcd1': str,       # Path to target point cloud (GeoTransformer)
    'src': str,        # Path to source point cloud (OverlapPredator)
    'tgt': str,        # Path to target point cloud (OverlapPredator)
    'rotation': np.ndarray(3,3),     # Ground truth rotation matrix
    'translation': np.ndarray(3,),   # Ground truth translation vector
    'scene_name': str,               # Scene identifier
    'frag_id0': int,                 # Source fragment ID
    'frag_id1': int,                 # Target fragment ID
}
```

## 2. Pylon Dataset Design Requirements

### 2.1 Core Structure
Every Pylon dataset must return three dictionaries:
1. **inputs**: Actual input data
2. **labels**: Ground truth for supervision
3. **meta_info**: Metadata (paths, indices, etc.)

### 2.2 Key Methods
- `_init_annotations()`: Initialize lightweight metadata list
- `_load_datapoint(idx)`: Load actual data on-demand
- Return format: `Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]`

### 2.3 Device Handling Rules
- Create tensors on CPU in `_load_datapoint()`
- Never use `.to(device)` in datasets
- BaseDataset handles device transfer automatically

## 3. Component Design

### 3.1 self.annotations Structure
```python
# Each annotation entry will be a dictionary:
{
    'src_path': str,           # Path to source point cloud file
    'tgt_path': str,           # Path to target point cloud file  
    'rotation': np.ndarray,    # Ground truth rotation (3,3)
    'translation': np.ndarray, # Ground truth translation (3,)
    'overlap': float,          # Overlap ratio
    'scene_name': str,         # Scene identifier
    'frag_id0': int,          # Source fragment ID
    'frag_id1': int,          # Target fragment ID
}
```

### 3.2 Inputs Dictionary
```python
{
    'src_pc': {
        'pos': torch.Tensor,   # Source positions (N, 3)
        'feat': torch.Tensor,  # Source features (N, D) - ones if no features
    },
    'tgt_pc': {
        'pos': torch.Tensor,   # Target positions (M, 3)  
        'feat': torch.Tensor,  # Target features (M, D) - ones if no features
    },
    'correspondences': torch.Tensor,  # Correspondence indices (K, 2)
}
```

### 3.3 Labels Dictionary
```python
{
    'transform': torch.Tensor,  # 4x4 transformation matrix
}
```

### 3.4 Meta Info Dictionary
```python
{
    'src_path': str,          # Source file path
    'tgt_path': str,          # Target file path
    'scene_name': str,        # Scene name
    'overlap': float,         # Overlap ratio
    'src_frame': int,         # Source frame ID
    'tgt_frame': int,         # Target frame ID
}
```

## 4. Data Loading Process

### 4.1 Point Cloud Loading
- Files are stored as PyTorch tensors (.pth files)
- Use `torch.load()` to load point clouds
- Expected format: `torch.Tensor` of shape (N, 3) for positions
- May include additional features/colors

### 4.2 Sampling Strategy
- Both references downsample to max_points (30000 default)
- Use random permutation for sampling: `np.random.permutation(N)[:max_points]`
- Deterministic per-index seeding required

### 4.3 Correspondence Generation
- Use existing `utils.point_cloud_ops.correspondences.get_correspondences`
- Parameters: matching_radius (overlap_radius in references)
- Apply transformation to source before finding correspondences

## 5. Transform Mapping

### 5.1 Already Implemented in Pylon
1. **RandomRigidTransform** (`data/transforms/vision_3d/random_rigid_transform.py`)
   - Handles rotation (rot_mag) and translation (trans_mag)
   - Supports both Rodrigues and Euler methods
   - Matches reference augmentation logic

2. **UniformPosNoise** (`data/transforms/vision_3d/uniform_pos_noise.py`)
   - Adds uniform noise to positions
   - Need to adapt for Gaussian noise

### 5.2 Transforms Needing Implementation
1. **GaussianPosNoise** - Add Gaussian noise to point positions
   - Reference uses: `(np.random.rand(N,3) - 0.5) * augment_noise`
   - Should implement proper Gaussian: `torch.randn(N,3) * noise_std`

### 5.3 Transform Composition
Use Pylon's Compose transform to chain:
1. RandomRigidTransform (if augment=True)
2. GaussianPosNoise (if augment=True)

## 6. Implementation Steps

### 6.1 Phase 1: Basic Structure
1. Inherit from BaseDataset
2. Define class attributes (SPLIT_OPTIONS, INPUT_NAMES, etc.)
3. Implement `__init__` with parameters matching references
4. Implement `_init_annotations()` to load metadata

### 6.2 Phase 2: Data Loading
1. Implement `_load_datapoint()` method
2. Load point clouds from .pth files
3. Apply downsampling if needed
4. Generate correspondences

### 6.3 Phase 3: Transforms
1. Implement GaussianPosNoise transform
2. Configure transform pipeline in __init__
3. Test augmentation behavior

### 6.4 Phase 4: Testing & Validation
1. Test data loading with actual files
2. Verify correspondence generation
3. Check augmentation effects
4. Validate against reference implementations

## 7. Key Differences from References

### 7.1 Structural Differences
- Pylon separates transforms from dataset logic
- Uses three-dictionary return format
- Handles device placement automatically
- Supports caching through BaseDataset

### 7.2 API Differences
- Reference returns tuple of tensors
- Pylon returns structured dictionaries
- Transform pipeline is configurable
- Seed handling is automatic per-index

## 8. Critical Implementation Notes

1. **Never manually handle device transfers** - trust BaseDataset
2. **Use local torch.Generator for randomness** - don't use global seeds
3. **Implement lazy loading** - only metadata in annotations
4. **Follow tensor type conventions** - float32 for positions, int64 for indices
5. **Handle missing data gracefully** - check file existence
6. **Support both 3DMatch and 3DLoMatch** - via overlap threshold parameter