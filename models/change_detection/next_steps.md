# Next Steps for 3D Point Cloud Change Detection Models

## Current Implementation Status
We currently have implemented:
- SiameseKPConv: A Siamese network based on KPConv for 3D point cloud change detection

## Recommended Models for Implementation

### Siamese KPConv: 3D multiple change detection from raw point clouds using deep learning
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing
- **Year**: 2023
- **Paper Link**: https://doi.org/10.1016/j.isprsjprs.2023.02.001
- **Citations**: 36
- **GitHub Repository**: https://github.com/IdeGelis/torch-points3d-SiameseKPConv
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
- **Key Features**:
  - Uses KPConv to process raw 3D point clouds
  - Employs a Siamese architecture for bi-temporal point cloud comparison
  - Achieves state-of-the-art performance on 3D change detection
- **Integration Notes**:
  - Core implementation already available in our codebase
  - Can serve as a baseline for other methods

### Change detection needs change information: Improving deep 3-d point cloud change detection
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing
- **Year**: 2024
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10415398
- **Citations**: 10
- **GitHub Repository**: https://github.com/IdeGelis/torch-points3d-SiamKPConvVariants
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
- **Key Features**:
  - Enhances SiameseKPConv with hand-crafted features, especially change-related ones
  - Proposes three new architectures: OneConvFusion, Triplet KPConv, and Encoder Fusion SiamKPConv
  - Improves IoU over classes of change by more than 5% compared to SoTA methods
- **Integration Notes**:
  - Extension of SiameseKPConv, so can build on existing implementation
  - Focuses on incorporating change information in early network stages

### DC3DCD: Unsupervised learning for multiclass 3D point cloud change detection
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing
- **Year**: 2023
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0924271623002976
- **Citations**: 4
- **GitHub Repository**: https://github.com/IdeGelis/torch-points3d-DC3DCD
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
- **Key Features**:
  - Unsupervised learning approach for 3D change detection
  - Eliminates the need for large labeled datasets
  - Handles multiclass change detection
- **Integration Notes**:
  - Complements SiameseKPConv with unsupervised learning capabilities
  - Valuable for scenarios with limited labeled data

### PGN3DCD: Prior-Knowledge-Guided Network for Urban 3D Point Cloud Change Detection
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing
- **Year**: 2024
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10620319
- **Citations**: 1
- **GitHub Repository**: https://github.com/zhanwenxiao/PGN3DCD
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
  - HKCD: A new dataset introduced by the authors
- **Key Features**:
  - Incorporates prior knowledge into the change detection process
  - Specifically designed for urban environments
  - Likely improves detection accuracy in complex urban scenes
- **Integration Notes**:
  - Can enhance existing models with prior knowledge incorporation
  - Potentially addresses challenges specific to urban point clouds

### An End-to-End Point-Based Method and a New Dataset for Street-Level Point Cloud Change Detection
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing
- **Year**: 2023
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10184135
- **Citations**: 7
- **GitHub Repository**: https://github.com/wangle53/3DCDNet
- **Benchmark Datasets**:
  - SLPCCD: Street-Level Point Cloud Change Detection (introduced by the authors)
  - Urb3DCD: Urban point cloud simulated dataset
- **Key Features**:
  - End-to-end point-based method for street-level change detection
  - Comes with a new dataset (SLPCCD) for street-level point clouds
  - Uses RandLA-Net architecture for feature extraction
- **Integration Notes**:
  - Focuses on street-level scenarios
  - Includes pretrained models and testing pipeline

### Point cloud registration and change detection in urban environment using an onboard lidar sensor and MLS reference data
- **Venue**: International Journal of Applied Earth Observation and Geoinformation
- **Year**: 2022
- **Paper Link**: https://doi.org/10.1016/j.jag.2022.102767
- **Citations**: Not specified
- **GitHub Repository**: No official repository (Some related work: https://github.com/JorgesNofulla/Point-Cloud-Urban-Change-detection)
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
  - Custom MLS dataset
- **Key Features**:
  - Addresses both registration and change detection in urban environments
  - Uses onboard LiDAR and Mobile Laser Scanning (MLS) reference data
  - Focuses on practical real-world urban applications
- **Integration Notes**:
  - Potentially useful for autonomous navigation applications
  - Handles the critical registration component, which is often a prerequisite for accurate change detection

### ChangeGAN: A deep network for change detection in coarsely registered point clouds
- **Venue**: IEEE Robotics and Automation Letters
- **Year**: 2021
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/9518246
- **Citations**: Multiple
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Synthetic urban dataset created by the authors
- **Key Features**:
  - Uses a GAN-based approach for handling coarsely registered point clouds
  - Addresses the challenge of registration errors in change detection
  - Particularly useful in robotics and autonomous navigation
- **Integration Notes**:
  - Complements existing methods by adding robustness to registration errors
  - Could improve performance in real-world scenarios with imperfect data alignment

### Street environment change detection from mobile laser scanning point clouds
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing
- **Year**: 2015
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0924271615001471
- **Citations**: High (frequently cited)
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom mobile laser scanning dataset from Paris
- **Key Features**:
  - Focuses specifically on street environment monitoring
  - Uses mobile laser scanning data
  - Includes semantic classification of changed objects
- **Integration Notes**:
  - Valuable for urban monitoring applications
  - Provides techniques for semantic understanding of changes

### Comparative analysis of machine learning and point-based algorithms for detecting 3D changes in buildings over time using bi-temporal lidar data
- **Venue**: Automation in Construction
- **Year**: 2019
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0926580518308963
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom bi-temporal LiDAR dataset of buildings
- **Key Features**:
  - Compares various approaches for 3D change detection
  - Combines point-based and machine learning techniques
  - Specifically designed for building change monitoring
- **Integration Notes**:
  - Offers multiple algorithmic approaches that could be integrated
  - Provides comparative insights valuable for implementation decisions

### LaserSAM: Zero-Shot Change Detection Using Visual Segmentation of Spinning LiDAR
- **Venue**: arXiv preprint
- **Year**: 2024
- **Paper Link**: https://arxiv.org/abs/2402.10321
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom dataset with spinning LiDAR sequences
- **Key Features**:
  - Zero-shot approach requiring no specific training for change detection
  - Leverages visual segmentation techniques for LiDAR data
  - Particularly applicable to spinning LiDAR setups
- **Integration Notes**:
  - Offers a different paradigm from learning-based approaches
  - Could be valuable for rapid deployment without extensive training

### Optimal Transport for Change Detection on LiDAR Point Clouds
- **Venue**: IEEE International Geoscience and Remote Sensing Symposium (IGARSS)
- **Year**: 2023
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10283101
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Urb3DCD: Urban point cloud simulated dataset
- **Key Features**:
  - Applies optimal transport theory to point cloud change detection
  - Likely offers mathematical rigor to the matching problem
  - Novel approach compared to purely deep learning methods
- **Integration Notes**:
  - Could complement learning-based approaches with theoretical guarantees
  - Potentially useful for specific applications requiring precise change localization

### Landslide Detection in 3D Point Clouds With Deep Siamese Convolutional Network
- **Venue**: IEEE International Geoscience and Remote Sensing Symposium (IGARSS)
- **Year**: 2024
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10641348
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom landslide LiDAR dataset
- **Key Features**:
  - Specializes in landslide detection from 3D point clouds
  - Adapts Siamese convolutional architecture for geological applications
  - Domain-specific application of change detection
- **Integration Notes**:
  - Demonstrates how to adapt general change detection to specific use cases
  - Valuable for geophysical applications

### A Change Detection Method for Misaligned Point Clouds in Mobile Robot System
- **Venue**: IEEE Conference
- **Year**: 2023
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/10473543
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom robotics-specific point cloud dataset
- **Key Features**:
  - Addresses the specific challenge of misalignment in point clouds
  - Designed for mobile robot systems
  - Handles practical issues in real-world deployments
- **Integration Notes**:
  - Could improve robustness of existing methods to alignment issues
  - Particularly relevant for robotics applications

### Construction scene change detection based on octree structure optimization
- **Venue**: SPIE Conference Proceedings
- **Year**: 2024
- **Paper Link**: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13402/1340211/Construction-scene-change-detection-based-on-octree-structure-optimization/10.1117/12.3049115.short
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom construction site point cloud dataset
- **Key Features**:
  - Uses octree data structures to optimize change detection
  - Specifically targets construction scenes
  - Likely offers computational efficiency improvements
- **Integration Notes**:
  - Could provide optimization techniques for existing methods
  - Valuable for construction monitoring applications

### Street-side vehicle detection, classification and change detection using mobile laser scanning data
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing
- **Year**: 2016
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0924271616000290
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom mobile laser scanning dataset from Paris
- **Key Features**:
  - Specialized in vehicle detection, classification, and change detection
  - Uses mobile laser scanning data
  - Combines object detection with change monitoring
- **Integration Notes**:
  - Useful for intelligent transportation systems
  - Provides specialized techniques for vehicle-focused applications

### Integrated change detection and classification in urban areas based on airborne laser scanning point clouds
- **Venue**: Sensors
- **Year**: 2018
- **Paper Link**: https://www.mdpi.com/1424-8220/18/2/448
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom airborne laser scanning dataset
- **Key Features**:
  - Integrates change detection with classification in a unified framework
  - Specifically designed for urban areas
  - Uses airborne laser scanning data
- **Integration Notes**:
  - Could enhance semantic understanding of detected changes
  - Valuable for urban planning and monitoring applications

### City-scale scene change detection using point clouds
- **Venue**: IEEE International Conference on Robotics and Automation
- **Year**: 2021
- **Paper Link**: https://ieeexplore.ieee.org/abstract/document/9561252
- **Citations**: Not specified
- **GitHub Repository**: https://github.com/ZJULiHongxin/HRNet-MSFA (related to approach)
- **Benchmark Datasets**:
  - Custom city-scale dataset collected by the authors
- **Key Features**:
  - Scales to city-level change detection
  - Likely offers efficient processing techniques for large-scale data
  - Aimed at practical deployment in real urban settings
- **Integration Notes**:
  - Could provide scalability improvements to existing methods
  - Valuable for large-area monitoring applications

### Change detection and deformation analysis in point clouds
- **Venue**: Photogrammetric Engineering & Remote Sensing
- **Year**: 2013
- **Paper Link**: https://www.ingentaconnect.com/content/asprs/pers/2013/00000079/00000005/art00004
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Several terrestrial laser scanning datasets
- **Key Features**:
  - Focuses on both change detection and deformation analysis
  - Likely provides techniques for monitoring subtle changes
  - Valuable for structural monitoring applications
- **Integration Notes**:
  - Could complement existing methods with deformation analysis capabilities
  - Useful for infrastructure monitoring and natural hazard assessment

### SHREC 2021: 3D point cloud change detection for street scenes
- **Venue**: Computers & Graphics
- **Year**: 2021
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0097849321001886
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - SHREC 2021 benchmark dataset for street-level change detection
- **Key Features**:
  - Benchmark dataset and competition for street scene change detection
  - Evaluates various methods on common data
  - Provides comparative analysis of different approaches
- **Integration Notes**:
  - Valuable for evaluating and comparing performance of different implementations
  - Could serve as an additional validation dataset

### Toward automated spatial change analysis of MEP components using 3D point clouds and as-designed BIM models
- **Venue**: Automation in Construction
- **Year**: 2019
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0926580518307908
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom building MEP component dataset
- **Key Features**:
  - Focuses on mechanical, electrical, and plumbing (MEP) components in buildings
  - Integrates BIM models with point clouds for change detection
  - Particularly useful for construction progress monitoring
- **Integration Notes**:
  - Specialized application for indoor construction environments
  - Could enhance the project's capabilities for building information modeling

### Change detection of urban trees in MLS point clouds using occupancy grids
- **Venue**: ISPRS Archives
- **Year**: 2019
- **Paper Link**: https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2-W13/1135/2019/
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom mobile laser scanning (MLS) dataset with trees
- **Key Features**:
  - Uses occupancy grids for efficient representation of point clouds
  - Specifically focuses on tree change detection in urban environments
  - Computationally efficient approach
- **Integration Notes**:
  - Could provide specialized vegetation monitoring capabilities
  - Useful for environmental and urban forestry applications

### 3D urban change detection with point cloud siamese networks
- **Venue**: Remote Sensing
- **Year**: 2020
- **Paper Link**: https://www.mdpi.com/2072-4292/12/24/4174
- **Citations**: Not specified
- **GitHub Repository**: https://github.com/grgzam/SiamVFE_SiamGCN-GCA
- **Benchmark Datasets**:
  - Custom urban point cloud datasets
- **Key Features**:
  - Applies Siamese network architecture specifically for urban change detection
  - Processes raw point clouds directly
  - Achieves good performance on urban datasets
- **Integration Notes**:
  - Could complement SiameseKPConv with different feature extraction approaches
  - Alternative Siamese network implementation for comparison

### Change detection for indoor construction progress monitoring based on BIM, point clouds and uncertainties
- **Venue**: Automation in Construction
- **Year**: 2020
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0926580519307782
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom indoor construction point cloud dataset
- **Key Features**:
  - Integrates Building Information Modeling (BIM) with point clouds
  - Accounts for uncertainties in measurements
  - Specifically designed for indoor construction environments
- **Integration Notes**:
  - Valuable for applications in construction progress monitoring
  - Could enhance the system's capabilities in dealing with uncertainty

### A voxel-based metadata structure for change detection in point clouds of large-scale urban areas
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing
- **Year**: 2019
- **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0924271619302175
- **Citations**: Not specified
- **GitHub Repository**: No official repository found
- **Benchmark Datasets**:
  - Custom large-scale urban point cloud dataset
- **Key Features**:
  - Proposes an efficient voxel-based metadata structure
  - Designed for handling large-scale urban point clouds
  - Improves computational efficiency in processing massive datasets
- **Integration Notes**:
  - Could significantly improve performance when dealing with large-scale data
  - Valuable for city-scale applications requiring efficient processing
