# Pylon Datasets

This directory contains documentation for all datasets available in Pylon.

## Complete Hierarchical Dataset List

### 1. Change Detection Datasets

#### 1.1 Bi-Temporal Change Detection
These datasets contain pairs of images or point clouds captured at two different time points for detecting changes.

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| AirChange | `AirChangeDataset` | 2D Image | [air_change.md](change_detection/bi_temporal/air_change.md) |
| CDD | `CDDDataset` | 2D Image | [cdd.md](change_detection/bi_temporal/cdd.md) |
| KC-3D | `KC3DDataset` | 3D Point Cloud | [kc_3d.md](change_detection/bi_temporal/kc_3d.md) |
| LEVIR-CD | `LevirCdDataset` | 2D Image | [levir_cd.md](change_detection/bi_temporal/levir_cd.md) |
| OSCD | `OSCDDataset` | 2D Image | [oscd.md](change_detection/bi_temporal/oscd.md) |
| SLPCCD | `SLPCCDDataset` | 3D Point Cloud | [slpccd.md](change_detection/bi_temporal/slpccd.md) |
| SYSU-CD | `SYSU_CD_Dataset` | 2D Image | [sysu_cd.md](change_detection/bi_temporal/sysu_cd.md) |
| URB3DCD | `Urb3DCDDataset` | 3D Point Cloud | [urb3dcd.md](change_detection/bi_temporal/urb3dcd.md) |
| xView2 | `xView2Dataset` | 2D Image | [xview2.md](change_detection/bi_temporal/xview2.md) |

#### 1.2 Single-Temporal Change Detection
These datasets are designed for change detection with a single temporal input or specialized processing.

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| Bi2SingleTemporal | `Bi2SingleTemporalDataset` | Wrapper | [bi2single_temporal.md](change_detection/single_temporal/bi2single_temporal.md) |
| I3PE | `I3PEDataset` | 2D Image | [i3pe.md](change_detection/single_temporal/i3pe.md) |
| PPSL | `PPSLDataset` | 2D Image | [ppsl.md](change_detection/single_temporal/ppsl.md) |

### 2. Multi-Task Learning Datasets
These datasets are designed for multi-task learning experiments with multiple objectives.

| Dataset | Class Name | Tasks | Documentation |
|---------|-----------|-------|---------------|
| ADE20K | `ADE20KDataset` | Segmentation | *Documentation needed* |
| CelebA | `CelebADataset` | Classification | [celeb_a.md](multi_task/celeb_a.md) |
| CityScapes | `CityScapesDataset` | Segmentation + Detection | [city_scapes.md](multi_task/city_scapes.md) |
| Multi-MNIST | `MultiMNISTDataset` | Multi-digit Classification | [multi_mnist.md](multi_task/multi_mnist.md) |
| Multi-Task Facial Landmark | `MultiTaskFacialLandmarkDataset` | Landmark Detection | [multi_task_facial_landmark.md](multi_task/multi_task_facial_landmark.md) |
| NYU-v2 | `NYUv2Dataset` | Depth + Segmentation | [nyu_v2.md](multi_task/nyu_v2.md) |
| PASCAL Context | `PascalContextDataset` | Segmentation + Detection | [pascal_context.md](multi_task/pascal_context.md) |

### 3. Point Cloud Registration (PCR) Datasets
These datasets are for 3D point cloud registration tasks.

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| BiTemporalPCR | `BiTemporalPCRDataset` | Wrapper | *Documentation needed* |
| KITTI | `KITTIDataset` | LiDAR | [kitti.md](pcr_datasets/kitti.md) |
| LidarCameraPosePCR | `LidarCameraPosePCRDataset` | LiDAR-Camera | *Documentation needed* |
| ModelNet40 | `ModelNet40Dataset` | Synthetic | *Documentation needed* |
| SingleTemporalPCR | `SingleTemporalPCRDataset` | Wrapper | *Documentation needed* |
| SyntheticTransformPCR | `SyntheticTransformPCRDataset` | Synthetic | *Documentation needed* |
| 3DMatch | `ThreeDMatchDataset` | RGB-D | [threedmatch.md](pcr_datasets/threedmatch.md) |

### 4. Semantic Segmentation Datasets

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| COCO-Stuff 164K | `COCOStuff164KDataset` | 2D Image | *Documentation needed* |
| WHU-BD | `WHUBDDataset` | 2D Image | *Documentation needed* |

### 5. Computer Vision Datasets (from torchvision)

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| MNIST | `MNIST` | 2D Image | *Uses torchvision wrapper* |

### 6. GAN Datasets

| Dataset | Class Name | Type | Documentation |
|---------|-----------|------|---------------|
| GAN Dataset | `GANDataset` | Synthetic | *Documentation needed* |

### 7. Random/Synthetic Datasets (for testing)

| Dataset | Class Name | Purpose | Documentation |
|---------|-----------|---------|---------------|
| ClassificationRandom | `ClassificationRandomDataset` | Testing | *Documentation needed* |
| SemanticSegmentationRandom | `SemanticSegmentationRandomDataset` | Testing | *Documentation needed* |

### 8. Special Dataset Wrappers

| Wrapper | Class Name | Purpose | Documentation |
|---------|-----------|---------|---------------|
| ProjectionDatasetWrapper | `ProjectionDatasetWrapper` | 3D->2D Projection | *Documentation needed* |

## Dataset Categories Summary

- **Total Datasets**: 40+
- **2D Image Datasets**: 20+
- **3D Point Cloud Datasets**: 10+
- **Multi-Task Datasets**: 7
- **Wrapper/Utility Datasets**: 5+

## Testing Coverage Requirements

When testing dataset functionality, ensure coverage for:
1. Cache version hash discrimination
2. Version dict functionality  
3. Data loading correctness
4. Soft link / relocatable dataset support
5. Cache metadata functionality
