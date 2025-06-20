# Environment Setup Guide <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. System Requirements](#2-system-requirements)
- [3. System Environment Setup](#3-system-environment-setup)
  - [3.1. G++ and GCC](#31-g-and-gcc)
  - [3.2. CUDA Toolkit 11.8](#32-cuda-toolkit-118)
- [4. Conda Environment Setup](#4-conda-environment-setup)
  - [4.1. Create conda environment](#41-create-conda-environment)
  - [4.2. Basics](#42-basics)
  - [4.3. Point cloud registration related](#43-point-cloud-registration-related)

## 1. Overview

This document outlines the setup process for the Pylon development environment. The environment is built using Conda for package management and includes dependencies for machine learning, computer vision, and related tools.

## 2. System Requirements

- Python 3.10
- CUDA 11.8 compatible GPU (for PyTorch GPU acceleration)

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

### 3.2. CUDA Toolkit 11.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

Add the following to `~/.bashrc`.
```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

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
conda install numpy==1.26.4 pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine mmcv==2.0.0 mmdet==3.0.0
pip install mmsegmentation==1.2.2
conda install -c conda-forge -y scipy scikit-learn scikit-image timm einops
conda install -c conda-forge -y opencv pycocotools rasterio imageio plyfile
pip install open3d laspy
conda install -c conda-forge -y matplotlib dash plotly pandas
conda install -c conda-forge -y psutil pytest tqdm
pip install rich
conda install -c conda-forge -y ftfy regex easydict
pip install fvcore triton jsbeautifier
```

### 4.3. Point cloud registration related

```bash
git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git && cd Pointnet2_PyTorch && pip install pointnet2_ops_lib/. && cd ..
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install ninja kornia einops easydict tensorboard tensorboardX nibabel
git clone https://github.com/KinglittleQ/torch-batch-svd.git && cd torch-batch-svd && python setup.py install && cd ..
```
