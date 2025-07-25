# Related Files List

This document lists all files in the repository related to:
- `data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset.py`
- `data/datasets/pcr_datasets/single_temporal_pcr_dataset.py` 
- `data/datasets/pcr_datasets/bi_temporal_pcr_dataset.py`
- `data/transforms/vision_3d/lidar_simulation_crop/`

## Core Dataset Files

- `/home/daniel/repos/Pylon-private/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset.py` - Main LiDAR camera pose dataset
- `/home/daniel/repos/Pylon-private/data/datasets/pcr_datasets/single_temporal_pcr_dataset.py` - Single temporal PCR dataset
- `/home/daniel/repos/Pylon-private/data/datasets/pcr_datasets/bi_temporal_pcr_dataset.py` - Bi-temporal PCR dataset
- `/home/daniel/repos/Pylon-private/data/datasets/pcr_datasets/synthetic_transform_pcr_dataset.py` - Parent abstract class

## LiDAR Simulation Crop Transform Files

- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/lidar_simulation_crop.py` - Main LiDAR simulation crop class
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/base_fov_crop.py` - Base FOV crop class
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/ellipsoid_fov_crop.py` - Ellipsoid FOV implementation
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/frustum_fov_crop.py` - Frustum FOV implementation
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/occlusion_crop.py` - Occlusion simulation
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/lidar_simulation_crop/range_crop.py` - Range-based filtering

## Related Transform Files

- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/pcr_translation.py` - PCR translation transform
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/__init__.py` - Module imports

## Configuration Files

### Training Configs
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/lidar_camera_pose_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/single_temporal_pcr_data_cfg.py` 
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/bi_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/geotransformer_single_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/geotransformer_bi_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/overlappredator_single_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/train/overlappredator_bi_temporal_pcr_data_cfg.py`

### Validation Configs
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/val/geotransformer_single_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/val/geotransformer_bi_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/val/overlappredator_single_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/val/overlappredator_bi_temporal_pcr_data_cfg.py`

### Evaluation Configs
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/eval/single_temporal_pcr_data_cfg.py`
- `/home/daniel/repos/Pylon-private/configs/common/datasets/point_cloud_registration/eval/bi_temporal_pcr_data_cfg.py`

## Benchmark Configurations

### Single Temporal PCR Benchmarks (0.4 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.4/TeaserPlusPlus_run_0.py`

### Single Temporal PCR Benchmarks (0.5 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_0.5/TeaserPlusPlus_run_0.py`

### Single Temporal PCR Benchmarks (1.0 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/single_temporal_pcr_1.0/TeaserPlusPlus_run_0.py`

### Bi-Temporal PCR Benchmarks (0.4 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.4/TeaserPlusPlus_run_0.py`

### Bi-Temporal PCR Benchmarks (0.5 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_0.5/TeaserPlusPlus_run_0.py`

### Bi-Temporal PCR Benchmarks (1.0 overlap)
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/GeoTransformer_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/GeoTransformer_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/GeoTransformer_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/ICP_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/OverlapPredator_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/OverlapPredator_run_1.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/OverlapPredator_run_2.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/RANSAC_FPFH_run_0.py`
- `/home/daniel/repos/Pylon-private/configs/benchmarks/point_cloud_registration/bi_temporal_pcr_1.0/TeaserPlusPlus_run_0.py`

## Test Files

### LiDAR Camera Pose PCR Dataset Tests
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/conftest.py`
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/test_datapoint_structure.py`
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/test_determinism.py`
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/test_initialization.py`
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/test_input_validation.py`
- `/home/daniel/repos/Pylon-private/tests/data/datasets/pcr_datasets/lidar_camera_pose_pcr_dataset/test_nerfstudio_format.py`

### Transform Tests
- `/home/daniel/repos/Pylon-private/tests/data/transforms/vision_3d/lidar_simulation_crop/test_lidar_simulation_crop.py`
- `/home/daniel/repos/Pylon-private/tests/data/transforms/vision_3d/test_pcr_translation.py`

## Demo and Visualization Files

- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/__init__.py`
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/main.py` - Demo entry point
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/README.md` - Demo documentation
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/webapp/__init__.py`
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/webapp/backend/__init__.py`
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/webapp/backend/visualization.py`
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/webapp/callbacks.py`
- `/home/daniel/repos/Pylon-private/demos/data/transforms/vision_3d/lidar_simulation_crop/webapp/layout.py`

## Module Import Files

- `/home/daniel/repos/Pylon-private/data/datasets/__init__.py` - Dataset imports
- `/home/daniel/repos/Pylon-private/data/transforms/vision_3d/__init__.py` - Transform module imports

## Summary

**Total: 102+ files** are related to these components across the entire repository, including:
- 4 core dataset implementation files
- 6 LiDAR simulation crop transform files  
- 2 related transform files
- 13 configuration files for training/validation/evaluation
- 54 benchmark configuration files across different overlap thresholds
- 8 test files
- 8 demo and visualization files
- 2 module import files

The components form a comprehensive point cloud registration system with synthetic data generation, LiDAR sensor simulation, and extensive benchmarking capabilities.