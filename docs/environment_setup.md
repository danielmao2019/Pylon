# Environment Setup Guide <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. System Requirements](#2-system-requirements)
- [3. Installation Steps](#3-installation-steps)
  - [3.1. Conda Environment Setup](#31-conda-environment-setup)
  - [3.2. Deep Learning Framework](#32-deep-learning-framework)
  - [3.3. Core Dependencies](#33-core-dependencies)
    - [3.3.1. Scientific and Machine Learning](#331-scientific-and-machine-learning)
    - [3.3.2. Computer Vision](#332-computer-vision)
    - [3.3.3. Visualization and Data Analysis](#333-visualization-and-data-analysis)
    - [3.3.4. System Utilities](#334-system-utilities)
    - [3.3.5. Development and Testing](#335-development-and-testing)
    - [3.3.6. Text and File Processing](#336-text-and-file-processing)

## 1. Overview

This document outlines the setup process for the Pylon development environment. The environment is built using Conda for package management and includes dependencies for machine learning, computer vision, and related tools.

## 2. System Requirements

- Python 3.10
- CUDA 11.8 compatible GPU (for PyTorch GPU acceleration)

## 3. Installation Steps

### 3.1. Conda Environment Setup

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

### 3.2. Deep Learning Framework

Install PyTorch and related packages:
```bash
# Install PyTorch with CUDA support
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install MMEngine and related components
pip install -U openmim
mim install mmengine mmcv==2.0.0 mmdet==3.0.0
pip install mmsegmentation==1.2.2
```

### 3.3. Core Dependencies

#### 3.3.1. Scientific and Machine Learning

```bash
conda install -c conda-forge -y \
    scipy \
    scikit-learn \
    scikit-image \
    timm \
    einops
```

#### 3.3.2. Computer Vision

```bash
conda install -c conda-forge -y \
    opencv \
    pycocotools \
    rasterio \
    imageio
```

#### 3.3.3. Visualization and Data Analysis

```bash
conda install -c conda-forge -y \
    matplotlib \
    dash \
    plotly \
    pandas \
    tqdm
```

#### 3.3.4. System Utilities

```bash
conda install -c conda-forge -y psutil
```

#### 3.3.5. Development and Testing

```bash
conda install -c conda-forge -y pytest
```

#### 3.3.6. Text and File Processing

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
