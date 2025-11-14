# Environment Setup Guide <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. System Requirements](#2-system-requirements)
- [3. System Environment Setup](#3-system-environment-setup)
  - [3.1. G++ and GCC](#31-g-and-gcc)
  - [3.2. CUDA Toolkit 12.8](#32-cuda-toolkit-128)
  - [3.3. COLMAP with GPU Support](#33-colmap-with-gpu-support)
- [4. Conda Environment Setup](#4-conda-environment-setup)
  - [4.1. Create conda environment](#41-create-conda-environment)
  - [4.2. Basics](#42-basics)
  - [4.3. Segmentation related](#43-segmentation-related)
  - [4.4. Point cloud registration related](#44-point-cloud-registration-related)
  - [4.5. Point Cloud Registration CUDA Extensions](#45-point-cloud-registration-cuda-extensions)

## 1. Overview

This document outlines the setup process for the Pylon development environment. The environment is built using Conda for package management and includes dependencies for machine learning, computer vision, and related tools.

## 2. System Requirements

- Python 3.10
- CUDA 12.8 compatible GPU (for PyTorch GPU acceleration)

## 3. System Environment Setup

### 3.1. G++ and GCC

```bash
sudo apt -y install gcc-9 g++-9
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

### 3.2. CUDA Toolkit 12.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

Add the following to `~/.bashrc`.
```bash
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 3.3. COLMAP with GPU Support

COLMAP is required for Structure-from-Motion in the NeRF Studio data generation pipeline. The system package doesn't have CUDA support, so we need to compile from source.

**Important**: The official COLMAP installation guide doesn't specify explicit boost library paths, which can cause library version conflicts. Our solution explicitly forces CMake to use system boost libraries.

```bash
# Install COLMAP dependencies (from official guide)
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Clean any previous installations to avoid conflicts
sudo rm -rf /usr/local/bin/colmap*
sudo rm -rf /usr/local/lib/colmap*
sudo rm -rf /usr/local/include/colmap*
sudo rm -rf /usr/local/share/colmap*
sudo ldconfig

# Clone and build COLMAP from source
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build

# CRITICAL: Unlike the official guide, we explicitly specify boost paths
# This prevents CMake from finding wrong boost versions (e.g., conda environments)
# and forces it to use system boost libraries to avoid runtime linking errors
cmake .. -GNinja \
  -DCMAKE_CUDA_COMPILER=`which nvcc` \
  -DBoost_ROOT=/usr \
  -DBoost_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu \
  -DBoost_INCLUDE_DIRS=/usr/include \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBoost_USE_STATIC_LIBS=OFF

ninja
sudo ninja install

# Verify installation has CUDA support
colmap --help | head -5
# Should show "COLMAP 3.7 -- Structure-from-Motion and Multi-View Stereo (with CUDA)"
```

**Changes from official guide:**
- Added explicit boost library paths (`-DBoost_ROOT`, `-DBoost_LIBRARY_DIRS`, `-DBoost_INCLUDE_DIRS`)
- Added `-DBoost_NO_SYSTEM_PATHS=ON` to prevent searching in conda/other environments
- Added cleanup commands to remove conflicting previous installations
- This prevents the common `libboost_program_options.so.1.78.0: cannot open shared object file` error

## 4. Conda Environment Setup

### 4.1. Create conda environment

First, ensure you have the latest version of Conda and create a new environment:
```bash
# Update Conda to latest version
conda update -n base -c defaults conda -y

# Create and activate new environment
conda create --name Pylon python=3.10 -y
conda activate Pylon

# Update pip to latest version
pip install --upgrade pip
```

### 4.2. Basics

```bash
# Install PyTorch 2.7.1 with CUDA 12.8
conda install numpy==1.26.4 -c conda-forge -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install compatible setuptools for CUDA extension compilation
pip install setuptools==59.5.0

# Install OpenMMLab packages (compatible versions)
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmdet==3.2.0 mmsegmentation==1.2.2

# Install other dependencies
conda install -c conda-forge -y scipy scikit-learn scikit-image timm einops
conda install -c conda-forge -y opencv pycocotools rasterio imageio plyfile
pip install open3d laspy
conda install -c conda-forge -y matplotlib dash plotly pandas psutil pytest tqdm ftfy regex easydict
pip install rich paramiko jsbeautifier fvcore triton xxhash
```

### 4.3. Segmentation related

```bash
pip install segmentation-models-pytorch@git+https://github.com/ragavsachdeva/segmentation_models.pytorch.git@2cde92e776b0a074d5e2f4f6a50c68754f948015
```

### 4.4. Point cloud registration related

```bash
# Install PyTorch3D from source to ensure CUDA compatibility
# Note: conda install pytorch3d breaks CUDA PyTorch, so we build from source
pip install --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install ninja kornia einops easydict tensorboard tensorboardX nibabel
git clone https://github.com/KinglittleQ/torch-batch-svd.git && cd torch-batch-svd && python setup.py install && cd ..
```

### 4.5. Point Cloud Registration CUDA Extensions

Compile the required CUDA extensions for point cloud registration models:

```bash
# GeoTransformer extensions
cd data/collators/geotransformer && python setup.py install && cd ../../..

# Buffer/OverlapPredator extensions  
cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..

# D3Feat extensions
cd data/collators/d3feat/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..

# OverlapPredator extensions
cd data/collators/overlappredator/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..

# PointNet2 operations for Buffer (avoids JIT compilation hanging)
cd models/point_cloud_registration/buffer/pointnet2_ops && python setup.py build_ext --inplace && cd ../../../..

# 1. PARENet CPU extensions (grid_subsampling, radius_neighbors)
cd models/point_cloud_registration/parenet && python setup.py build_ext --inplace && cd ../../..

# 2. PARENet CUDA extensions (pointops)
cd models/point_cloud_registration/parenet/pareconv/extensions/pointops && python setup.py install && cd ../../../../../..
```
