#!/usr/bin/env bash
# Creates and populates the Pylon conda environment.
# Assumes system setup (setup_system.sh) has already been run and the VM has
# been rebooted so that the NVIDIA driver and CUDA toolkit are active.
#
# Required env var:
#   PYLON_REPO_DIR — absolute path to the repo root.
#
# Usage:
#   PYLON_REPO_DIR=~/repos/sthetic-face-prod bash docs/env_setup/setup_pylon_env.sh
set -euo pipefail

assert_var() { [ -n "${!1:-}" ] || { echo "ERROR: $1 is not set."; exit 1; }; }
assert_var PYLON_REPO_DIR

export PATH="/usr/local/cuda-11.8/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

echo "=== [1/5] Create Pylon conda env ==="
if ! conda env list | grep -q "^Pylon "; then
    conda create --name Pylon python=3.10 -y
fi
conda activate Pylon
pip install --upgrade pip

echo "=== [2/5] Pip requirements ==="
pip install -r "$PYLON_REPO_DIR/docs/env_setup/requirements-torch-cu118.txt"
# pytorch3d needs --no-build-isolation (setup.py imports torch at top level)
pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@stable'
# Install remaining extras (skip git+ packages already handled or needing post-reboot GPU)
grep -v '^git+\|^#\|^$' "$PYLON_REPO_DIR/docs/env_setup/requirements-extras.txt" | \
    pip install -r /dev/stdin \
    --constraint "$PYLON_REPO_DIR/docs/env_setup/requirements-torch-cu118.txt"

echo "=== [3/5] OpenMMLab packages ==="
pip install "setuptools<76"
mim install --no-deps mmengine mmdet==3.2.0 mmsegmentation==1.2.2
pip install --no-build-isolation mmcv==2.1.0

echo "=== [4/5] Source-only packages ==="
cd /tmp
if [ -d torch-batch-svd ]; then rm -rf torch-batch-svd; fi
git clone https://github.com/KinglittleQ/torch-batch-svd.git
cd torch-batch-svd && python setup.py install && cd ..
rm -rf torch-batch-svd

echo "=== [5/5] C++ / CUDA extensions ==="
# geotransformer
cd "$PYLON_REPO_DIR/data/collators/geotransformer"
python setup.py install

# collator cpp extensions
CPP_DIRS=(
    "$PYLON_REPO_DIR/data/collators/overlappredator/cpp_wrappers/cpp_neighbors"
    "$PYLON_REPO_DIR/data/collators/overlappredator/cpp_wrappers/cpp_subsampling"
    "$PYLON_REPO_DIR/data/collators/d3feat/cpp_wrappers/cpp_neighbors"
    "$PYLON_REPO_DIR/data/collators/d3feat/cpp_wrappers/cpp_subsampling"
    "$PYLON_REPO_DIR/data/collators/buffer/cpp_wrappers/cpp_neighbors"
    "$PYLON_REPO_DIR/data/collators/buffer/cpp_wrappers/cpp_subsampling"
)
for dir in "${CPP_DIRS[@]}"; do
    echo "Building: $dir"
    cd "$dir"
    python setup.py build_ext --inplace
done

# pointnet2_ops
echo "Building: pointnet2_ops"
cd "$PYLON_REPO_DIR/models/point_cloud_registration/buffer/pointnet2_ops"
python setup.py build_ext --inplace

# parenet
echo "Building: parenet"
cd "$PYLON_REPO_DIR/models/point_cloud_registration/parenet"
python setup.py build_ext --inplace

# parenet pointops
echo "Building: parenet pointops"
cd "$PYLON_REPO_DIR/models/point_cloud_registration/parenet/pareconv/extensions/pointops"
python setup.py install

conda deactivate

echo ""
echo "Pylon environment setup complete."
