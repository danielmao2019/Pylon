# 3DMatch Dataset Family

## Overview

The 3DMatch dataset family consists of real-world RGB-D scans from **91 indoor scenes** designed for point cloud registration evaluation. It provides two complementary benchmarks:

- **3DMatch**: Standard registration benchmark with mixed overlap distribution
- **3DLoMatch**: Low-overlap registration benchmark for challenging scenarios

Both datasets use identical scene splits but different overlap-based filtering strategies.

## Quick Start

### Basic Usage
```python
from data.datasets import ThreeDMatchDataset, ThreeDLoMatchDataset

# Standard 3DMatch dataset (overlap > 0.3)
dataset = ThreeDMatchDataset(
    data_root='./data/datasets/soft_links/threedmatch',
    split='train',
    matching_radius=0.1,
)

# Low-overlap 3DLoMatch dataset (0.1 < overlap ≤ 0.3)
lomatch_dataset = ThreeDLoMatchDataset(
    data_root='./data/datasets/soft_links/threedmatch',
    split='train',
    matching_radius=0.1,
)

print(f"3DMatch train pairs: {len(dataset)}")        # 14,313 pairs
print(f"3DLoMatch train pairs: {len(lomatch_dataset)}")  # 6,225 pairs
```

### Custom Overlap Range
```python
from data.datasets.pcr_datasets.threedmatch_dataset import _ThreeDMatchBaseDataset

# Custom overlap range dataset
custom_dataset = _ThreeDMatchBaseDataset(
    data_root='./data/datasets/soft_links/threedmatch',
    split='train',
    overlap_min=0.5,    # 50% minimum overlap
    overlap_max=0.8,    # 80% maximum overlap
    matching_radius=0.1,
)
```

## Dataset Statistics

### Dataset Sizes Summary

| Dataset | Train | Val | Test | Overlap Filter |
|---------|-------|-----|------|----------------|
| **3DMatch** | 14,313 pairs | 915 pairs | 1,520 pairs | overlap > 0.3 |
| **3DLoMatch** | 6,225 pairs | 414 pairs | 1,772 pairs | 0.1 < overlap ≤ 0.3 |

### Metadata Files Overview

| File | Total Pairs | Description |
|------|-------------|-------------|
| **train.pkl** | 20,642 | All training pairs from 75 scenes |
| **val.pkl** | 1,331 | All validation pairs from 8 scenes |
| **3DMatch.pkl** | 1,623 | Pre-curated test pairs (mixed overlap) |
| **3DLoMatch.pkl** | 1,781 | Pre-curated test pairs (low overlap) |

## Dataset Design

### Overlap-Based Partitioning Strategy

**Train/Val Splits**: Perfect overlap-based partitioning
- Source: Same metadata files (`train.pkl`, `val.pkl`)
- Method: Runtime filtering by overlap range
- Result: 0 pairs overlap between 3DMatch and 3DLoMatch

**Test Splits**: Pre-curated evaluation sets
- Source: Different metadata files (`3DMatch.pkl`, `3DLoMatch.pkl`)
- Method: Independent curation with different philosophies
- Result: 103 pairs from 3DMatch.pkl match 3DLoMatch filter (6.8% of 3DMatch filtered pairs)

### Why Test Sets Overlap

The 103 overlapping test pairs exist because:
1. **Independent curation**: Files created separately by different researchers
2. **Boundary cases**: Both include pairs in the 0.1-0.3 overlap range
3. **Evaluation philosophy**: Both wanted challenging borderline cases
4. **Minimal impact**: Only 6.8% overlap doesn't significantly affect evaluation validity

## Scene Splits

### Training: 75 Scenes
<details>
<summary>Click to expand scene list</summary>

```
7-scenes-chess, 7-scenes-fire, 7-scenes-office, 7-scenes-pumpkin, 7-scenes-stairs,
analysis-by-synthesis-apt1-kitchen, analysis-by-synthesis-apt1-living,
analysis-by-synthesis-apt2-bed, analysis-by-synthesis-apt2-kitchen,
analysis-by-synthesis-apt2-living, analysis-by-synthesis-apt2-luke,
analysis-by-synthesis-office2-5a, analysis-by-synthesis-office2-5b,
bundlefusion-apt0_1, bundlefusion-apt0_2, bundlefusion-apt0_3, bundlefusion-apt0_4,
bundlefusion-apt1_1, bundlefusion-apt1_2, bundlefusion-apt1_3, bundlefusion-apt1_4,
bundlefusion-apt2_1, bundlefusion-apt2_2, bundlefusion-copyroom_1, bundlefusion-copyroom_2,
bundlefusion-office1_1, bundlefusion-office1_2, bundlefusion-office2, bundlefusion-office3,
rgbd-scenes-v2-scene_01 through rgbd-scenes-v2-scene_14 (excluding 10),
sun3d-brown_bm_1-brown_bm_1_1, sun3d-brown_bm_1-brown_bm_1_2, sun3d-brown_bm_1-brown_bm_1_3,
sun3d-brown_cogsci_1-brown_cogsci_1, sun3d-brown_cs_2-brown_cs2_1, sun3d-brown_cs_2-brown_cs2_2,
sun3d-brown_cs_3-brown_cs3, sun3d-harvard_c3-hv_c3_1, sun3d-harvard_c5-hv_c5_1,
sun3d-harvard_c6-hv_c6_1, sun3d-harvard_c8-hv_c8_3, sun3d-hotel_nips2012-nips_4_1,
sun3d-hotel_nips2012-nips_4_2, sun3d-hotel_sf-scan1_1 through sun3d-hotel_sf-scan1_4,
sun3d-mit_32_d507-d507_2_1, sun3d-mit_32_d507-d507_2_2,
sun3d-mit_46_ted_lab1-ted_lab_2_1 through sun3d-mit_46_ted_lab1-ted_lab_2_4,
sun3d-mit_76_417-76-417b_1 through sun3d-mit_76_417-76-417b_5,
sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika,
sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_1 through _4
```
</details>

### Validation: 8 Scenes
```
7-scenes-heads, analysis-by-synthesis-apt2-kitchen, bundlefusion-office0_1,
bundlefusion-office0_2, bundlefusion-office0_3, rgbd-scenes-v2-scene_10,
sun3d-brown_bm_4-brown_bm_4, sun3d-harvard_c11-hv_c11_2
```

### Test: 8 Scenes
```
7-scenes-redkitchen, sun3d-home_at-home_at_scan1_2013_jan_1,
sun3d-home_md-home_md_scan9_2012_sep_30, sun3d-hotel_uc-scan3,
sun3d-hotel_umd-maryland_hotel1, sun3d-hotel_umd-maryland_hotel3,
sun3d-mit_76_studyroom-76-1studyroom2, sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

## ⚠️ Known Issues

### Data Leakage Between Train/Val

**Problem**: Scene `analysis-by-synthesis-apt2-kitchen` appears in BOTH training and validation splits.

**Impact**:
- Scene appears in both train and validation metadata
- May result in data leakage during validation
- Exact pair counts depend on overlap filtering applied
- May inflate validation performance metrics

**Mitigation**:
1. Remove leaked scene from validation (use only 7 clean scenes)
2. Report results both with and without leaked scene
3. Use test set for final evaluation only

## Data Format

### Input/Output Structure
```python
from data.structures.three_d.point_cloud.point_cloud import PointCloud

{
    'inputs': {
        'src_pc': PointCloud(
            xyz=src_positions,                # Source point cloud (M, 3)
            data={'feat': src_feat}            # Features aligned with src_pc.num_points
        ),
        'tgt_pc': PointCloud(
            xyz=tgt_positions,                # Target point cloud (N, 3)
            data={'feat': tgt_feat}            # Features aligned with tgt_pc.num_points
        ),
        'correspondences': torch.Tensor,      # Point correspondences (K, 2)
    },
    'labels': {
        'transform': torch.Tensor              # 4x4 transformation matrix
    },
    'meta_info': {
        'idx': int,                           # Dataset index
        'src_path': str,                      # Source point cloud path
        'tgt_path': str,                      # Target point cloud path
        'scene_name': str,                    # Scene name
        'overlap': float,                     # Overlap ratio
        'src_frame': int,                     # Source fragment ID
        'tgt_frame': int,                     # Target fragment ID
    }
}
```

`PointCloud` guarantees non-empty coordinate tensors and keeps positional (`xyz`) data synchronized with optional fields, so downstream code should never reintroduce dictionary-based fallbacks or revalidate emptiness manually. Use `pc.num_points` whenever you need to check the number of points instead of peeking at `xyz.shape[0]`, and keep every optional field aligned with that cardinality.

### File System Structure
```
data_root/
├── metadata/
│   ├── train.pkl       # Training pairs metadata
│   ├── val.pkl         # Validation pairs metadata
│   ├── 3DMatch.pkl     # 3DMatch test pairs
│   └── 3DLoMatch.pkl   # 3DLoMatch test pairs
├── train/
│   └── {scene_name}/
│       ├── cloud_bin_{id}.pth
│       └── cloud_bin_{id}.info.txt
└── test/
    └── {scene_name}/
        ├── cloud_bin_{id}.pth
        └── cloud_bin_{id}.info.txt
```

## Implementation Details

### Metadata Format
All `.pkl` files use GeoTransformer format (list of dictionaries):
```python
[
    {
        'pcd0': str,              # Source point cloud path
        'pcd1': str,              # Target point cloud path
        'rotation': np.ndarray,   # (3, 3) rotation matrix
        'translation': np.ndarray,# (3,) translation vector
        'overlap': float,         # Overlap ratio [0, 1]
        'scene_name': str,        # Scene identifier
        'frag_id0': int,          # Source fragment ID
        'frag_id1': int,          # Target fragment ID
    },
    ...
]
```

### Filtering Logic
```python
# Runtime overlap filtering
if self.overlap_min < overlap <= self.overlap_max:
    # Include pair in dataset

# 3DMatch: overlap_min=0.3, overlap_max=1.0
# 3DLoMatch: overlap_min=0.1, overlap_max=0.3
```

### Key Features
- **Correspondence caching**: Computed correspondences cached for performance
- **Thread-safe**: Supports multi-worker DataLoader
- **Device handling**: Automatic GPU/CPU transfer handled by BaseDataset
- **Scene validation**: Ensures source/target from same scene

## Detailed Statistics

<details>
<summary>Overlap Distribution Analysis</summary>

### Training Set (20,642 pairs)
| Overlap Range | Pairs | Percentage |
|---------------|-------|------------|
| < 0.1 | 104 | 0.5% |
| 0.1-0.3 | 6,225 | 30.2% |
| 0.3-0.5 | 6,415 | 31.1% |
| 0.5-0.7 | 4,956 | 24.0% |
| > 0.7 | 2,942 | 14.3% |

### Validation Set (1,331 pairs)
| Overlap Range | Pairs | Percentage |
|---------------|-------|------------|
| < 0.1 | 2 | 0.2% |
| 0.1-0.3 | 414 | 31.1% |
| 0.3-0.5 | 390 | 29.3% |
| 0.5-0.7 | 326 | 24.5% |
| > 0.7 | 199 | 14.9% |

### Test Sets
- **3DMatch**: 1,623 total pairs -> 1,520 filtered pairs (93.7% have overlap > 0.3)
- **3DLoMatch**: 1,781 total pairs -> 1,772 filtered pairs (99.5% have 0.1 < overlap ≤ 0.3)
- **Cross-dataset overlap**: 103 pairs from 3DMatch.pkl match 3DLoMatch filter range

</details>

## References

1. **3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions** (CVPR 2017)
   - Original dataset paper
   - Established standard evaluation protocol

2. **PREDATOR: Registration of 3D Point Clouds with Low Overlap** (CVPR 2021)
   - Introduced 3DLoMatch variant
   - Emphasized low-overlap challenges

3. **GeoTransformer: Fast and Robust Point Cloud Registration** (CVPR 2022)
   - Used both benchmarks
   - State-of-the-art results
