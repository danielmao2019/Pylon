# 3DMatch Dataset

## Dataset Description

The 3DMatch dataset is a collection of real-world RGB-D scans of indoor scenes, specifically designed for geometric registration and matching. It contains point cloud fragments from various indoor environments like apartments, offices, and hotel rooms. The dataset is particularly useful for evaluating point cloud registration algorithms in real-world scenarios with challenging factors like occlusions, sensor noise, and varying scene complexity.

## File System Structure

The dataset should be organized as follows:

```
root_dir/
├── metadata/
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
├── data/
│   ├── scene1/
│   │   ├── cloud_bin_0.npz
│   │   ├── cloud_bin_1.npz
│   │   └── ...
│   ├── scene2/
│   └── ...
└── fragments/
    ├── scene1/
    │   ├── cloud_bin_0.ply
    │   ├── cloud_bin_1.ply
    │   └── ...
    ├── scene2/
    └── ...
```

## Data Structure

Each data point consists of a pair of point cloud fragments with their relative transformation. The data is structured as follows:

### Point Cloud Fragment (.npz file)
- `points`: (N, 3) float32 array of 3D points
- `features`: (N, D) float32 array of per-point features (if available)
- `normals`: (N, 3) float32 array of per-point normals (if available)

### Metadata (.pkl file)
Each entry in the metadata file contains:
- `scene_name`: str, name of the scene
- `frag_id0`: str, ID of the first fragment
- `frag_id1`: str, ID of the second fragment
- `overlap`: float, overlap ratio between fragments
- `rotation`: (3, 3) float32 array, ground truth rotation matrix
- `translation`: (3,) float32 array, ground truth translation vector

### Dataset Output
The dataset class returns a dictionary containing:
- `ref_points`: (N, 3) float32 array, reference point cloud
- `src_points`: (N, 3) float32 array, source point cloud
- `rotation`: (3, 3) float32 array, ground truth rotation
- `translation`: (3,) float32 array, ground truth translation
- `scene_name`: str, name of the scene
- `ref_frame`: str, reference frame ID
- `src_frame`: str, source frame ID
- `overlap`: float, overlap ratio

## Download Instructions

1. Download the 3DMatch dataset from the official website:
   ```bash
   wget http://3dmatch.cs.princeton.edu/3DMatch.tar.gz
   ```

2. Extract the downloaded file:
   ```bash
   tar -xzf 3DMatch.tar.gz
   ```

3. Process the raw data to generate fragments and features:
   ```bash
   # Install required dependencies
   pip install open3d numpy torch

   # Use provided scripts to process data
   python scripts/process_3dmatch.py --input_dir /path/to/3DMatch --output_dir /path/to/processed
   ```

## Usage in Research Works

The 3DMatch dataset has been widely used in point cloud registration research:

1. **3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions** (CVPR 2017)
   - First introduced the dataset
   - Used for learning local geometric descriptors

2. **Deep Closest Point: Learning Representations for Point Cloud Registration** (ICCV 2019)
   - Evaluated end-to-end point cloud registration
   - Demonstrated robustness to real-world noise and occlusions

3. **PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency** (CVPR 2021)
   - Used for evaluating spatial consistency in registration
   - Achieved state-of-the-art performance on challenging indoor scenes

4. **GeoTransformer: A Geometric Transformer for 3D Point Cloud Registration** (CVPR 2022)
   - Demonstrated effectiveness of transformer architecture
   - Showed improved performance on partial overlap cases

5. **PREDATOR: Registration of 3D Point Clouds with Low Overlap** (CVPR 2021)
   - Focused on low-overlap scenarios
   - Introduced overlap prediction module 