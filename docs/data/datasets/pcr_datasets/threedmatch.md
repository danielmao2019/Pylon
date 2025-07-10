# 3DMatch Dataset Family

## Overview

The 3DMatch dataset is a collection of real-world RGB-D scans from **91 indoor scenes**, designed for point cloud registration and geometric matching evaluation. The dataset provides two complementary benchmarks: **3DMatch** (standard evaluation) and **3DLoMatch** (low-overlap evaluation), both using identical scene splits but different overlap characteristics.

## Dataset Variants

### 3DMatch Dataset
**Official 3DMatch dataset for standard point cloud registration evaluation.**
- **Purpose**: Standard point cloud registration benchmark
- **Test pairs**: 1,623 pairs with mixed overlap distribution
- **Overlap statistics**: Mean 0.50, range 0.14-0.96
- **Evaluation**: Standard registration scenarios

### 3DLoMatch Dataset  
**3DLoMatch considers only scan pairs with low overlaps for challenging evaluation.**
- **Purpose**: Challenging low-overlap registration benchmark
- **Test pairs**: 1,781 pairs with low overlap distribution
- **Overlap statistics**: Mean 0.19, range 0.10-0.30 (99.5% < 0.3)
- **Evaluation**: Challenging registration scenarios where traditional methods often fail

## Scene-Based Data Splits

Both 3DMatch and 3DLoMatch use **identical scene-based splits** with no overlap between train/validation/test scenes.

### Training Split (75 scenes)
**75 unique scenes used for model training:**

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
rgbd-scenes-v2-scene_01, rgbd-scenes-v2-scene_02, rgbd-scenes-v2-scene_03,
rgbd-scenes-v2-scene_04, rgbd-scenes-v2-scene_05, rgbd-scenes-v2-scene_06,
rgbd-scenes-v2-scene_07, rgbd-scenes-v2-scene_08, rgbd-scenes-v2-scene_09,
rgbd-scenes-v2-scene_11, rgbd-scenes-v2-scene_12, rgbd-scenes-v2-scene_13, rgbd-scenes-v2-scene_14,
sun3d-brown_bm_1-brown_bm_1_1, sun3d-brown_bm_1-brown_bm_1_2, sun3d-brown_bm_1-brown_bm_1_3,
sun3d-brown_cogsci_1-brown_cogsci_1, sun3d-brown_cs_2-brown_cs2_1, sun3d-brown_cs_2-brown_cs2_2,
sun3d-brown_cs_3-brown_cs3, sun3d-harvard_c3-hv_c3_1, sun3d-harvard_c5-hv_c5_1,
sun3d-harvard_c6-hv_c6_1, sun3d-harvard_c8-hv_c8_3, sun3d-hotel_nips2012-nips_4_1,
sun3d-hotel_nips2012-nips_4_2, sun3d-hotel_sf-scan1_1, sun3d-hotel_sf-scan1_2,
sun3d-hotel_sf-scan1_3, sun3d-hotel_sf-scan1_4, sun3d-mit_32_d507-d507_2_1,
sun3d-mit_32_d507-d507_2_2, sun3d-mit_46_ted_lab1-ted_lab_2_1, sun3d-mit_46_ted_lab1-ted_lab_2_2,
sun3d-mit_46_ted_lab1-ted_lab_2_3, sun3d-mit_46_ted_lab1-ted_lab_2_4, sun3d-mit_76_417-76-417b_1,
sun3d-mit_76_417-76-417b_2_1, sun3d-mit_76_417-76-417b_3, sun3d-mit_76_417-76-417b_4,
sun3d-mit_76_417-76-417b_5, sun3d-mit_dorm_next_sj-dorm_next_sj_oct_30_2012_scan1_erika,
sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_1, 
sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_2,
sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_3,
sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_4
```

### Validation Split (8 scenes)
**8 unique scenes used for hyperparameter tuning:**

```
7-scenes-heads, analysis-by-synthesis-apt2-kitchen, bundlefusion-office0_1,
bundlefusion-office0_2, bundlefusion-office0_3, rgbd-scenes-v2-scene_10,
sun3d-brown_bm_4-brown_bm_4, sun3d-harvard_c11-hv_c11_2
```

⚠️ **CRITICAL: Data Leakage Detected**

**Problem**: Scene `analysis-by-synthesis-apt2-kitchen` appears in BOTH training and validation splits.

**Impact**: This constitutes a serious data leakage issue that can artificially inflate validation performance metrics, leading to:
- **Overoptimistic validation scores** during hyperparameter tuning
- **Poor generalization** to truly unseen test data  
- **Invalid model selection** based on contaminated validation results
- **Unreliable cross-validation** for algorithm comparison

**Quantitative Analysis**:
- **Leaked scene**: `analysis-by-synthesis-apt2-kitchen`
- **Training pairs from leaked scene**: 75 pairs
- **Validation pairs from leaked scene**: 75 pairs  
- **Total contaminated pairs**: 150 pairs
- **Validation contamination rate**: 75/1331 = 5.6% of validation set
- **Training contamination rate**: 75/20642 = 0.4% of training set

**Root Cause**: This appears to be an oversight in the original dataset split creation, where scene-level splits were not properly enforced between training and validation sets.

**Mitigation Strategies**:
1. **Remove leaked scene from validation**: Use only 7 clean validation scenes
2. **Alternative validation**: Use clean test scenes for validation (with separate held-out test set)
3. **Cross-validation**: Use scene-based k-fold splits ensuring no scene overlap
4. **Report both**: Show results with and without leaked scene for transparency

**Research Impact**: Any validation-based hyperparameter tuning or model selection using this split may have inflated performance estimates. Results should be interpreted cautiously and confirmed on truly independent test sets.

### Test Split (8 scenes)
**8 unique scenes used for final evaluation:**

```
7-scenes-redkitchen, sun3d-home_at-home_at_scan1_2013_jan_1,
sun3d-home_md-home_md_scan9_2012_sep_30, sun3d-hotel_uc-scan3,
sun3d-hotel_umd-maryland_hotel1, sun3d-hotel_umd-maryland_hotel3,
sun3d-mit_76_studyroom-76-1studyroom2, sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

## Dataset Sizes and Statistics

### Raw Dataset Statistics (Before Filtering)

| Split | Scenes | Total Pairs | Overlap Range | Mean Overlap |
|-------|--------|-------------|---------------|--------------|
| Train | 75     | **20,642**  | 0.0006-0.9985 | 0.44         |
| Val   | 8      | **1,331**   | 0.0957-0.9871 | 0.44         |
| Test  | 8      | **3,304***  | 0.0986-0.9614 | 0.33         |

*Combined test universe from both 3DMatch and 3DLoMatch metadata files

### 3DMatch Dataset (Standard Evaluation)

| Split | Total Pairs | Filtered Pairs | Overlap Characteristics |
|-------|-------------|----------------|-------------------------|
| Train | 20,642      | **14,313** | 69.3% with overlap > 0.3 |
| Val   | 1,331       | **915** | 68.7% with overlap > 0.3 |
| Test  | 3,304*      | **1,623** | 49.1% with overlap > 0.3 (high-overlap partition) |

### 3DLoMatch Dataset (Low-Overlap Evaluation)

| Split | Total Pairs | Filtered Pairs | Overlap Characteristics |
|-------|-------------|----------------|-------------------------|
| Train | 20,642      | **6,225** | 30.2% with 0.1 < overlap ≤ 0.3 |
| Val   | 1,331       | **414** | 31.1% with 0.1 < overlap ≤ 0.3 |
| Test  | 3,304*      | **1,781** | 53.9% with overlap ≤ 0.3 (low-overlap partition) |

**Overlap-Based Partitioning**: 3DMatch and 3DLoMatch use the same underlying test pair universe but apply different overlap filters:
- **3DMatch**: Primarily high-overlap pairs (99.7% coverage of overlap > 0.3)
- **3DLoMatch**: Primarily low-overlap pairs (99.8% coverage of overlap ≤ 0.3)  
- **Small overlap**: 100 shared pairs (3.0% of total test universe)

*Total test universe: 3,304 unique pairs from 8 test scenes

### Key Statistics Summary

- **Universe**: 91 unique scenes (75 train + 8 val + 8 test)
- **Total pairs**: 25,177 across all splits
- **Test scene distribution**: Same 8 scenes for both 3DMatch and 3DLoMatch
- **Largest test scene**: `7-scenes-redkitchen` (506 pairs, 31% of 3DMatch test)

## Dataset Relationship

3DMatch and 3DLoMatch are **complementary evaluation protocols** using the same underlying data:

- **Same scenes**: Both use identical scene splits (91 total scenes)
- **Same train/val data**: Both use the same training and validation pairs
- **Different test filtering**: 3DMatch uses mixed overlaps, 3DLoMatch uses low overlaps only
- **Complementary evaluation**: Together they provide comprehensive registration evaluation

### Test Set Partitioning Strategy

**Total test universe: 3,304 unique pairs from 8 test scenes**

3DMatch and 3DLoMatch apply **overlap-based partitioning** to this test universe:

| Partition | Pairs | Coverage | Purpose |
|-----------|-------|----------|---------|
| **3DMatch** (High overlap) | 1,623 | 99.7% of pairs with overlap > 0.3 | Standard registration evaluation |
| **3DLoMatch** (Low overlap) | 1,781 | 99.8% of pairs with overlap ≤ 0.3 | Challenging registration evaluation |
| **Shared pairs** | 100 | 3.0% overlap between datasets | Low-overlap pairs in both sets |

**Why different sizes?** The 8 test scenes naturally contain more low-overlap pairs (1,779) than high-overlap pairs (1,525), resulting in 3DLoMatch having more test pairs than 3DMatch.

This partitioning allows comprehensive evaluation across the overlap spectrum using the same scene geometry.

## Complete Dataset Statistics

### Complete Filtered Pair Counts

| Dataset | Train Pairs | Val Pairs | Test Pairs | Overlap Filter | Purpose |
|---------|-------------|-----------|------------|----------------|---------|
| **3DMatch** | **14,313** / 20,642 (69.3%) | **915** / 1,331 (68.7%) | **1,623** / 3,304 (49.1%) | overlap > 0.3 | Standard registration |
| **3DLoMatch** | **6,225** / 20,642 (30.2%) | **414** / 1,331 (31.1%) | **1,781** / 3,304 (53.9%) | 0.1 < overlap ≤ 0.3 | Low-overlap registration |

**Consistent Partitioning**: All splits (train/val/test) use the same overlap-based filtering philosophy, making 3DMatch and 3DLoMatch true dataset variants rather than just different test protocols.

### Overlap Threshold Analysis

**Training Set (20,642 total pairs)**:
| Threshold | Pairs ≥ Threshold | Pairs < Threshold | Percentage ≥ |
|-----------|-------------------|-------------------|---------------|
| ≥ 0.1 | 20,538 pairs | 104 pairs | 99.5% |
| ≥ 0.2 | 17,901 pairs | 2,741 pairs | 86.7% |
| ≥ 0.3 | 14,313 pairs | 6,329 pairs | 69.3% |
| ≥ 0.4 | 10,983 pairs | 9,659 pairs | 53.2% |
| ≥ 0.5 | 7,898 pairs | 12,744 pairs | 38.3% |

**Validation Set (1,331 total pairs)**:
| Threshold | Pairs ≥ Threshold | Pairs < Threshold | Percentage ≥ |
|-----------|-------------------|-------------------|---------------|
| ≥ 0.1 | 1,329 pairs | 2 pairs | 99.8% |
| ≥ 0.2 | 1,133 pairs | 198 pairs | 85.1% |
| ≥ 0.3 | 915 pairs | 416 pairs | 68.7% |
| ≥ 0.4 | 707 pairs | 624 pairs | 53.1% |
| ≥ 0.5 | 525 pairs | 806 pairs | 39.4% |

## File System Structure

```
data_root/
├── metadata/
│   ├── train.pkl       # Training pairs metadata (20,642 pairs)
│   ├── val.pkl         # Validation pairs metadata (1,331 pairs)
│   └── 3DMatch.pkl     # Test pairs metadata (1,623 pairs)
└── data/
    ├── train/
    │   └── test_scene/     # Limited test data
    │       ├── cloud_bin_0.pth
    │       ├── cloud_bin_1.pth
    │       └── ...
    └── test/
        └── (various scenes)/
            ├── cloud_bin_0.pth
            ├── cloud_bin_1.pth
            └── ...
```

## Data Structure

### Point Cloud Files (.pth)
Each point cloud is stored as a PyTorch tensor:
- **Format**: `torch.Tensor` of shape `(N, 3)` 
- **Type**: `float32`
- **Content**: 3D coordinates of point cloud

### Metadata Structure (.pkl)
Each metadata file contains:
```python
{
    'src': List[str],        # Source point cloud paths
    'tgt': List[str],        # Target point cloud paths  
    'rot': np.ndarray,       # Rotation matrices (N, 3, 3)
    'trans': np.ndarray,     # Translation vectors (N, 3)
    'overlap': np.ndarray,   # Overlap ratios (N,)
}
```

### Dataset Output Format
Both `ThreeDMatchDataset` and `ThreeDLoMatchDataset` return:
```python
{
    'inputs': {
        'src_pc': {
            'pos': torch.Tensor,    # Source point cloud (M, 3)
            'feat': torch.Tensor,   # Features (M, 1) - all ones
        },
        'tgt_pc': {
            'pos': torch.Tensor,    # Target point cloud (N, 3)
            'feat': torch.Tensor,   # Features (N, 1) - all ones
        },
        'correspondences': torch.Tensor,  # Point correspondences (K, 2)
    },
    'labels': {
        'transform': torch.Tensor,        # 4x4 transformation matrix
    },
    'meta_info': {
        'idx': int,                      # Dataset index
        'src_path': str,                 # Source point cloud path
        'tgt_path': str,                 # Target point cloud path
        'scene_name': str,               # Scene name
        'overlap': float,                # Overlap ratio
        'src_frame': int,                # Source fragment ID
        'tgt_frame': int,                # Target fragment ID
    }
}
```

## Usage Examples

### Basic Usage
```python
from data.datasets import ThreeDMatchDataset, ThreeDLoMatchDataset

# Standard 3DMatch dataset (mixed overlaps)
dataset = ThreeDMatchDataset(
    data_root='./data/datasets/soft_links/threedmatch',
    split='train',
    matching_radius=0.1,
)

# Low overlap 3DLoMatch dataset  
lomatch_dataset = ThreeDLoMatchDataset(
    data_root='./data/datasets/soft_links/threedmatch',
    split='train',
    matching_radius=0.1,
)

print(f"3DMatch train pairs: {len(dataset)}")        # 14,313 pairs (overlap > 0.3)
print(f"3DLoMatch train pairs: {len(lomatch_dataset)}")  # 6,225 pairs (0.1 < overlap ≤ 0.3)
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

## Research Papers

1. **3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions** (CVPR 2017)
   - Original paper introducing the dataset
   - Established the standard evaluation protocol

2. **PREDATOR: Registration of 3D Point Clouds with Low Overlap** (CVPR 2021)  
   - Introduced 3DLoMatch for low-overlap evaluation
   - Demonstrated importance of challenging overlap scenarios

3. **GeoTransformer: Fast and Robust Point Cloud Registration via Geometric Transformer** (CVPR 2022)
   - Used both 3DMatch and 3DLoMatch for comprehensive evaluation
   - Showed transformer effectiveness across overlap ranges

## Implementation Notes

- **Filtering**: Both datasets filter at runtime using `overlap_min < overlap <= overlap_max`
- **Test evaluation**: 3DMatch uses 1,623 test pairs, 3DLoMatch uses 1,781 test pairs  
- **Train/Val consistency**: Both datasets use identical training and validation data
- **Correspondence caching**: Correspondences are cached in sibling directory for performance
- **Scene validation**: Automatic validation ensures source and target belong to same scene
- **Device handling**: Tensors created on CPU, BaseDataset handles device transfer
- **Thread safety**: All operations are thread-safe for multi-worker DataLoader usage