# Environment Setup Guide

## Overview
This document outlines the setup process for the Pylon development environment. The environment is built using Conda for package management and includes dependencies for machine learning, computer vision, and related tools.

## System Requirements
- Python 3.10
- CUDA 11.8 compatible GPU (for PyTorch GPU acceleration)

## Installation Steps

### 1. Conda Environment Setup
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

### 2. Deep Learning Framework
Install PyTorch and related packages:
```bash
# Install PyTorch with CUDA support
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install MMEngine and related components
pip install -U openmim
mim install mmengine mmcv==2.0.0 mmdet==3.0.0
pip install mmsegmentation==1.2.2
```

### 3. Core Dependencies

#### Scientific and Machine Learning
```bash
conda install -c conda-forge -y \
    scipy \
    scikit-learn \
    scikit-image \
    timm \
    einops
```

#### Computer Vision
```bash
conda install -c conda-forge -y \
    opencv \
    pycocotools \
    rasterio \
    imageio
```

#### Visualization and Data Analysis
```bash
conda install -c conda-forge -y \
    matplotlib \
    dash \
    plotly \
    pandas \
    tqdm
```

#### System Utilities
```bash
conda install -c conda-forge -y psutil
```

#### Development and Testing
```bash
conda install -c conda-forge -y pytest
```

#### Text and File Processing
```bash
# Conda packages
conda install -c conda-forge -y \
    ftfy \
    regex \
    plyfile

# Pip packages
pip install \
    fvcore \
    triton \
    jsbeautifier
```
