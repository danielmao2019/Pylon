# Conda Environment Setup

This directory contains all files needed for traditional conda environment setup.

## Quick Start

### 1. Manual Installation (Step-by-Step)
Follow the detailed guide in [`conda.md`](conda.md) for manual setup with explanations.

### 2. Automated Installation (Recommended)
Use the installation script for automated setup:

```bash
# Create conda environment
conda create --name pylon python=3.10 -y
conda activate pylon

# Run automated installation script
bash install_packages.sh
```

### 3. Environment File Installation
Use the environment specification file:

```bash
# Create environment from specification
conda env create -f environment.yml
conda activate pylon

# Build C++ extensions (if in Pylon directory)
cd data/collators/geotransformer && python setup.py install && cd ../../..
cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
cd data/collators/overlappredator/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
```

## Files Overview

- **[`conda.md`](conda.md)** - Detailed manual setup guide with system requirements
- **[`install_packages.sh`](install_packages.sh)** - Automated installation script
- **[`environment.yml`](environment.yml)** - Conda environment specification file

## System Requirements

- **Python 3.10**
- **CUDA 11.8** compatible GPU
- **GCC/G++ 9** for C++ extensions
- **Git** for repository cloning during setup

## Usage Notes

- The installation script **requires an active conda environment**
- C++ extensions are built during installation (GeoTransformer, Buffer, OverlapPredator)
- External repositories (Pointnet2_PyTorch, torch-batch-svd) are cloned and installed automatically
- All installations are logged with colored output for easy debugging

## Troubleshooting

**Common Issues:**
- **"No conda environment activated"**: Run `conda activate pylon` before the script
- **C++ compilation errors**: Ensure GCC 9 is installed and set as default
- **CUDA not found**: Verify CUDA 11.8 installation and PATH variables
- **Permission errors**: Run with appropriate permissions or use `sudo` for system packages

**Verification:**
After installation, verify with:
```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

For additional help, see the main [environment setup guide](../README.md).