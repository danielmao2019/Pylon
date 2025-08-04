#!/bin/bash
# Pylon Package Installation Script
# This script installs all required packages for Pylon in a conda environment
# Usage: bash install_packages.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

# Check if running in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    error "No conda environment activated. Please activate the Pylon environment first."
fi

log "Installing packages in conda environment: $CONDA_DEFAULT_ENV"

# Update pip to latest version
log "Updating pip..."
pip install --upgrade pip

# Install PyTorch and core ML libraries
log "Installing PyTorch and CUDA libraries..."
conda install -y numpy==1.26.4 pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install OpenMMLab stack
log "Installing OpenMMLab stack..."
pip install -U openmim
mim install mmengine mmcv==2.0.0 mmdet==3.0.0
pip install mmsegmentation==1.2.2

# Install scientific computing libraries
log "Installing scientific computing libraries..."
conda install -c conda-forge -y \
    scipy scikit-learn scikit-image timm einops \
    opencv pycocotools rasterio imageio plyfile \
    matplotlib dash plotly pandas psutil pytest tqdm ftfy regex easydict

# Install additional Python packages
log "Installing additional Python packages..."
pip install open3d laspy rich paramiko jsbeautifier fvcore triton

# Install segmentation models
log "Installing segmentation models..."
pip install "segmentation-models-pytorch@git+https://github.com/ragavsachdeva/segmentation_models.pytorch.git@2cde92e776b0a074d5e2f4f6a50c68754f948015"

# Install point cloud registration dependencies
log "Installing point cloud registration dependencies..."
conda install -y pytorch3d -c pytorch3d --freeze-installed
pip install ninja kornia einops easydict tensorboard tensorboardX nibabel

# Install KNN_CUDA
log "Installing KNN_CUDA..."
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Clone and install Pointnet2_PyTorch
log "Installing Pointnet2_PyTorch..."
if [ ! -d "/tmp/Pointnet2_PyTorch" ]; then
    git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/Pointnet2_PyTorch
fi
cd /tmp/Pointnet2_PyTorch && pip install pointnet2_ops_lib/. && cd -

# Clone and install torch-batch-svd
log "Installing torch-batch-svd..."
if [ ! -d "/tmp/torch-batch-svd" ]; then
    git clone https://github.com/KinglittleQ/torch-batch-svd.git /tmp/torch-batch-svd
fi
cd /tmp/torch-batch-svd && python setup.py install && cd -

# Build C++ extensions if in Pylon directory
if [ -f "data/collators/geotransformer/setup.py" ]; then
    log "Building GeoTransformer C++ extensions..."
    cd data/collators/geotransformer && python setup.py install && cd ../../..
fi

if [ -f "data/collators/buffer/cpp_wrappers/compile_wrappers.sh" ]; then
    log "Building Buffer C++ extensions..."
    cd data/collators/buffer/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
fi

if [ -f "data/collators/overlappredator/cpp_wrappers/compile_wrappers.sh" ]; then
    log "Building OverlapPredator C++ extensions..."
    cd data/collators/overlappredator/cpp_wrappers && bash compile_wrappers.sh && cd ../../../..
fi

# Cleanup temporary directories
log "Cleaning up temporary files..."
rm -rf /tmp/Pointnet2_PyTorch /tmp/torch-batch-svd

# Verify installation
log "Verifying installation..."
python -c "
import torch
import mmdet
import open3d
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('MMDetection version:', mmdet.__version__)
print('Open3D version:', open3d.__version__)
print('Installation successful!')
" || error "Installation verification failed"

log "✅ All packages installed successfully!"
log "You can now run Pylon experiments with: python main.py --config-filepath configs/examples/linear/config.py --debug"
