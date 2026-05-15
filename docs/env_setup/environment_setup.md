# Environment Setup Guide <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. Overview](#1-overview)
- [2. System Requirements](#2-system-requirements)
- [3. System Environment Setup](#3-system-environment-setup)
  - [3.1. Linux (NVIDIA/CUDA) setup](#31-linux-nvidiacuda-setup)
  - [3.2. macOS (Apple Silicon/MPS) setup](#32-macos-apple-siliconmps-setup)
  - [3.3. COLMAP with GPU Support (Linux only)](#33-colmap-with-gpu-support-linux-only)
- [4. Conda Environment Setup](#4-conda-environment-setup)
  - [4.1. Create conda environment](#41-create-conda-environment)
  - [4.2. Basics](#42-basics)
  - [4.3. Segmentation related](#43-segmentation-related)
  - [4.4. Point cloud registration related](#44-point-cloud-registration-related)
  - [4.5. Point Cloud Registration CUDA Extensions (Linux/NVIDIA only)](#45-point-cloud-registration-cuda-extensions-linuxnvidia-only)

## 1. Overview

This document outlines the setup process for the Pylon development environment. The environment is built using Conda for package management and includes dependencies for machine learning, computer vision, and related tools.

### Automated scripts

Two scripts automate the full setup described in sections 3 and 4:

```bash
# 1. System-level provisioning (packages, gcc-9, NVIDIA driver, CUDA 11.8, Miniconda).
#    Reboot after this step to activate the NVIDIA driver.
bash docs/env_setup/setup_system.sh

# 2. Pylon conda environment (pip packages, OpenMMLab, C++ extensions).
PYLON_REPO_DIR=$(pwd) bash docs/env_setup/setup_pylon_env.sh
```

For Linux, treat those scripts as the canonical command source. This guide intentionally avoids repeating commands that already live in the scripts. When you need to inspect or change the actual command sequence, edit the scripts or the requirements files they call instead of copying shell snippets into this document.

## 2. System Requirements

- Python 3.10
- Linux path: CUDA 11.8 compatible GPU (for PyTorch GPU acceleration)
- macOS path: Apple Silicon GPU via PyTorch MPS backend (no CUDA)

## 3. System Environment Setup

### 3.1. Linux (NVIDIA/CUDA) setup

Use `bash docs/env_setup/setup_system.sh`.

That script is the source of truth for Linux system provisioning. It covers:

1. system packages;
2. `gcc-9` / `g++-9` installation and selection;
3. NVIDIA driver installation;
4. CUDA 11.8 toolkit installation;
5. CUDA path export setup in `~/.bashrc`;
6. Miniconda installation and `conda init`.

If you need to change any Linux system-level command, update [setup_system.sh](setup_system.sh) rather than duplicating the command here.

### 3.2. macOS (Apple Silicon/MPS) setup

Install Xcode Command Line Tools first:

```bash
xcode-select --install
```

PyTorch on macOS uses the MPS backend (Metal); do not install CUDA toolkit on macOS.

### 3.3. COLMAP with GPU Support (Linux only)

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

Linux: this is covered by `PYLON_REPO_DIR=$(pwd) bash docs/env_setup/setup_pylon_env.sh`.

That script is the source of truth for the Linux `Pylon` env lifecycle. It covers:

1. `conda` shell hook activation;
2. environment creation;
3. activation of `Pylon`;
4. pip bootstrap;
5. Python package installation;
6. OpenMMLab installation;
7. source-only package installation;
8. point-cloud-registration extension builds.

If you need to change any Linux conda-environment command, update [setup_pylon_env.sh](setup_pylon_env.sh) or the requirements files it consumes rather than duplicating the command here.

macOS: create and activate the environment manually before following section 4.2.

### 4.2. Basics

Linux (NVIDIA/CUDA):

Covered by [setup_pylon_env.sh](setup_pylon_env.sh). The Linux package sources of truth are:

1. [requirements-torch-cu118.txt](requirements-torch-cu118.txt) for the pinned PyTorch stack;
2. [requirements-extras.txt](requirements-extras.txt) for the pinned extra Python dependencies;
3. [setup_pylon_env.sh](setup_pylon_env.sh) for the install order, including the separate `pytorch3d` install with `--no-build-isolation`.

macOS (Apple Silicon/MPS):

```bash
# Install the pinned Python dependencies
pip install --upgrade pip
pip install -r docs/env_setup/requirements-torch-macos.txt
pip install -r docs/env_setup/requirements-extras-macos.txt --constraint docs/env_setup/requirements-torch-macos.txt
```

If you also need Nerfstudio on macOS, install it separately after Xcode Command Line Tools are fully set up:

```bash
xcode-select --install
pip install nerfstudio==1.1.5
```

```bash
# Install OpenMMLab packages (not included in the requirements files)
pip install "setuptools<76"
mim install --no-deps mmengine mmdet==3.2.0 mmsegmentation==1.2.2
pip install --no-build-isolation mmcv==2.1.0
```

On macOS, OpenMMLab wheel availability is limited and may require source builds. Skip this block unless explicitly needed for your task.

### 4.3. Segmentation related

Linux: covered by [requirements-extras.txt](requirements-extras.txt) and installed by [setup_pylon_env.sh](setup_pylon_env.sh).

macOS: install the equivalent dependency set from [requirements-extras-macos.txt](requirements-extras-macos.txt).

### 4.4. Point cloud registration related

Linux:

1. `KNN_CUDA` is declared in [requirements-extras.txt](requirements-extras.txt).
2. `torch-batch-svd` is installed by [setup_pylon_env.sh](setup_pylon_env.sh) in the source-only packages phase.

On macOS, skip `KNN_CUDA` because CUDA is unavailable.

### 4.5. Point Cloud Registration CUDA Extensions (Linux/NVIDIA only)

Covered by [setup_pylon_env.sh](setup_pylon_env.sh). That script builds:

1. GeoTransformer extensions;
2. OverlapPredator collator extensions;
3. D3Feat collator extensions;
4. Buffer collator extensions;
5. Buffer `pointnet2_ops`;
6. PARENet CPU extensions;
7. PARENet CUDA `pointops`.

If you need to change the build commands or the extension list, update [setup_pylon_env.sh](setup_pylon_env.sh) rather than duplicating the shell commands here.
