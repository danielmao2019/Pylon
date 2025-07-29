# KITTI Dataset

## Dataset Description

The KITTI dataset is a large-scale dataset for autonomous driving research, containing LiDAR scans, stereo images, and GPS/IMU data collected from a moving vehicle in urban environments. For point cloud registration tasks, we primarily use the odometry benchmark part of the dataset, which provides sequences of LiDAR scans with ground truth poses.

## File System Structure

The dataset should be organized following the KITTI odometry benchmark structure:

```
root_dir/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/           # LiDAR point clouds
│   │   │   ├── 000000.bin
│   │   │   ├── 000001.bin
│   │   │   └── ...
│   │   └── calib.txt          # Calibration data
│   ├── 01/
│   └── ...
└── poses/                     # Ground truth poses
    ├── 00.txt
    ├── 01.txt
    └── ...
```

## Data Structure

### Point Cloud (.bin file)
- Binary file containing Nx4 float32 array
- Each point has format: (x, y, z, intensity)
- Coordinates are in vehicle coordinate system
- Points are ordered row by row

### Pose File (.txt)
- Each line contains 12 numbers representing a 3x4 transformation matrix
- Format: [R11 R12 R13 t1 R21 R22 R23 t2 R31 R32 R33 t3]
- Transformation is from vehicle coordinate system to world coordinate system

### Calibration File (calib.txt)
Contains various calibration matrices:
- P0/P1/P2/P3: Camera projection matrices
- Tr: Transformation from velodyne to camera coordinate system
- Tr_imu_to_velo: Transformation from IMU to velodyne coordinate system

## Download Instructions

1. Download the KITTI odometry benchmark from the official website:
   ```bash
   wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip  # Point clouds (80GB)
   wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip     # Ground truth poses
   wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip     # Calibration files
   ```

2. Extract the downloaded files:
   ```bash
   unzip data_odometry_velodyne.zip
   unzip data_odometry_poses.zip
   unzip data_odometry_calib.zip
   ```

3. Organize the files according to the file system structure shown above.

## Usage in Research Works

The KITTI dataset has been extensively used in point cloud registration research:

1. **DeepVCP: An End-to-End Deep Neural Network for Point Cloud Registration** (ICCV 2019)
   - Evaluated on outdoor LiDAR scans
   - Demonstrated robustness to sparse point clouds

2. **Deep Global Registration** (CVPR 2020)
   - Used KITTI for large-scale outdoor registration
   - Showed effectiveness in handling sequential point cloud pairs

3. **PREDATOR: Registration of 3D Point Clouds with Low Overlap** (CVPR 2021)
   - Evaluated on challenging outdoor scenarios
   - Demonstrated robustness to varying overlap ratios

4. **RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Features** (ICCV 2021)
   - Used KITTI for evaluating orientation-aware features
   - Showed improved performance on sequential registration

5. **SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration** (CVPR 2022)
   - Evaluated spatial compatibility on outdoor scenes
   - Demonstrated state-of-the-art performance on KITTI

## Dataset Split

The dataset is typically split as follows:
- Training: Sequences 00-05
- Validation: Sequences 06-07
- Testing: Sequences 08-10

This split ensures that the evaluation is performed on different environments and driving scenarios than those used for training.
