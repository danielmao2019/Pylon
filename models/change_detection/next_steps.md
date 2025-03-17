# Next Steps for 3D Point Cloud Change Detection Models

## Current Implementation Status
We currently have implemented:
- SiameseKPConv: A Siamese network based on KPConv for 3D point cloud change detection

## Recommended Models for Implementation

### 1. SiamGCN (Siamese Graph Convolutional Network)
- **Paper**: "3D Urban Change Detection with Point Cloud Siamese Networks" (ISPRS 2021) by Tao Ku et al.
- **Implementation Resources**:
  - Original code: https://github.com/kutao207/SiamGCN
- **Benchmark Dataset**:
  - SHREC2021: Street-level point cloud data with annotated changes
  - Contains both real LiDAR data and synthetic city scenes
- **Key Features**:
  - Uses EdgeConv operators to extract representative features from point clouds
  - Employs a Siamese architecture based on graph convolutional networks
  - Excellent for urban scene analysis and street-level change detection
- **Integration Notes**:
  - Would complement SiameseKPConv with graph-based feature learning
  - Particularly effective for sparse point cloud data

### 2. SiamVFE and SiamGCN-GCA
- **Paper**: Work related to "SiamVFE & SiamGCN-GCA for the task of point cloud change detection for city scenes"
- **Implementation Resources**:
  - Official repository: https://github.com/grgzam/SiamVFE_SiamGCN-GCA
  - Features pre-processing tools and evaluation scripts for city scenes
- **Benchmark Dataset**:
  - SHREC2023: Updated version of the street-level point cloud dataset
  - Contains both real LiDAR data (2016/2020) and synthetic city scenes
  - Available through the repository
- **Key Features**:
  - SiamVFE: Uses voxel feature encoding for point cloud change detection
  - SiamGCN-GCA: Enhances SiamGCN with Graph Context Attention mechanisms
  - Improved performance over the original SiamGCN architecture
- **Integration Notes**:
  - More advanced than original SiamGCN with additional attention mechanisms
  - Provides both voxel-based and graph-based approaches for comparison

### 3. ChangeGAN
- **Paper**: "ChangeGAN: A Deep Network for Change Detection in Coarsely Registered Point Clouds" (IEEE Robotics and Automation Letters, Vol. 6, 2021)
- **Implementation Resources**:
  - No public official implementation found
  - Can reference related GAN-based approaches for point clouds
- **Benchmark Dataset**:
  - Typically evaluated on custom datasets with deliberately misaligned point clouds
  - Could be tested on modified versions of Urb3DCD with artificial registration errors
- **Key Features**:
  - Generative adversarial network for point cloud change detection
  - Robust to registration errors and alignment issues
  - Works with coarsely registered point clouds
- **Integration Notes**:
  - Would add capability to handle imperfectly aligned point cloud data
  - GAN-based approach offers a different learning paradigm compared to existing models

### 4. Encoder Fusion SiamKPConv
- **Paper**: "Change detection needs change information: improving deep 3D point cloud change detection" (2023)
- **Implementation Resources**:
  - Official repository: https://github.com/IdeGelis/torch-points3d-SiamKPConvVariants
  - Contains implementations of multiple SiamKPConv variants:
    - OneConvFusion
    - Encoder Fusion SiamKPConv
    - Triplet KPConv
  - Built on top of the Torch-Points3D framework
- **Benchmark Dataset**:
  - Urb3DCD: Urban point cloud simulated dataset with various qualities
  - Includes low/high resolution with different noise levels
  - Publicly available at: https://ieee-dataport.org/open-access/urb3dcd-urban-point-clouds-simulated-dataset-3d-change-detection
- **Key Features**:
  - Improves upon SiamKPConv by incorporating change information in early network stages
  - Outperforms previous state-of-the-art by more than 5% on mean IoU for change classes
  - Better fusion of temporal information
- **Integration Notes**:
  - Could be implemented as an extension/improvement to our existing SiameseKPConv
