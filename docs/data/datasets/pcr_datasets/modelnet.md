# ModelNet40 Dataset

## Dataset Description

The ModelNet40 dataset is a large-scale 3D CAD model dataset containing 12,311 models from 40 object categories. For point cloud registration tasks, we use a preprocessed version where the 3D models are converted to point clouds. This dataset is particularly useful for evaluating registration algorithms on clean, synthetic data with well-defined object structures.

## File System Structure

The dataset should be organized as follows:

```
root_dir/
├── shape_names.txt           # List of category names
├── train_files.txt          # List of training files
├── test_files.txt           # List of testing files
├── modelnet40_ply_hdf5_2048/
│   ├── ply_data_train0.h5  # Training data
│   ├── ply_data_train1.h5
│   ├── ...
│   ├── ply_data_test0.h5   # Testing data
│   └── ...
└── modelnet40_normal_resampled/  # Optional: version with normal vectors
    ├── airplane/
    │   ├── train/
    │   └── test/
    ├── bathtub/
    └── ...
```

## Data Structure

### HDF5 Files (.h5)
Each HDF5 file contains:
- `data`: (N, 2048, 3) float32 array of point clouds
- `label`: (N,) int64 array of category labels
- `normal`: (N, 2048, 3) float32 array of normal vectors (if available)

### Dataset Output
The dataset class returns a dictionary containing:
- `src_points`: (N, 3) float32 array, source point cloud
- `tgt_points`: (N, 3) float32 array, target point cloud
- `transform`: (4, 4) float32 array, ground truth transformation
- `category`: str, object category name
- `category_id`: int, category index
- Optional fields if using normals:
  - `src_normals`: (N, 3) float32 array, source normals
  - `tgt_normals`: (N, 3) float32 array, target normals

## Categories

The dataset includes 40 object categories:
```python
ALL_CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]
```

## Download Instructions

1. Download the ModelNet40 dataset:
   ```bash
   # Download point cloud version (HDF5 format)
   wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

   # Optional: Download version with normal vectors
   wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
   ```

2. Extract the downloaded files:
   ```bash
   unzip modelnet40_ply_hdf5_2048.zip
   unzip modelnet40_normal_resampled.zip  # Optional
   ```

3. Organize the files according to the file system structure shown above.

## Usage in Research Works

The ModelNet40 dataset has been widely used in point cloud registration research:

1. **PointNetLK: Robust & Efficient Point Cloud Registration using PointNet** (CVPR 2019)
   - First to use ModelNet40 for learning-based registration
   - Demonstrated effectiveness on clean CAD models

2. **Deep Closest Point: Learning Representations for Point Cloud Registration** (ICCV 2019)
   - Used ModelNet40 for synthetic experiments
   - Showed robustness to different initial poses

3. **RPM-Net: Robust Point Matching using Learned Features** (CVPR 2020)
   - Evaluated on partial-to-partial registration
   - Demonstrated handling of outliers and noise

4. **Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration** (CVPR 2020)
   - Used ModelNet40 for feature learning
   - Showed effectiveness of semi-supervised training

5. **DeepGMR: Learning Latent Gaussian Mixture Models for Registration** (ECCV 2020)
   - Evaluated on clean and noisy versions
   - Demonstrated probabilistic registration approach

## Dataset Characteristics

1. **Clean Data**
   - CAD models provide clean, noise-free point clouds
   - Useful for initial algorithm development and testing

2. **Complete Objects**
   - Each point cloud represents a complete object
   - Good for evaluating global registration methods

3. **Synthetic Nature**
   - May not reflect real-world challenges
   - Often used in combination with real-world datasets

4. **Category Information**
   - Enables category-specific analysis
   - Useful for transfer learning experiments

5. **Normal Vectors**
   - Available in the normal_resampled version
   - Helpful for geometric feature extraction 