#!/usr/bin/env bash
# System-level provisioning for a bare GCP Ubuntu 22.04 GPU VM.
# Installs system packages, gcc-9, NVIDIA driver, CUDA 11.8, and Miniconda.
#
# After completion, reboot the VM to activate the NVIDIA driver.
#
# Usage:
#   bash docs/env_setup/setup_system.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/6] System packages ==="
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential git curl wget unzip tmux python3-venv

echo "=== [2/6] gcc-9 / g++-9 ==="
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gcc-9 g++-9
sudo update-alternatives --remove-all gcc 2>/dev/null || true
sudo update-alternatives --remove-all g++ 2>/dev/null || true
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --set gcc /usr/bin/gcc-9
sudo update-alternatives --set g++ /usr/bin/g++-9
echo "gcc: $(gcc --version | head -1)"
echo "g++: $(g++ --version | head -1)"

echo "=== [3/6] NVIDIA driver ==="
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install nvidia-driver-535
echo "nvidia-driver-535 installed (reboot required to activate)."

echo "=== [4/6] CUDA 11.8 toolkit ==="
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
rm -f cuda_11.8.0_520.61.05_linux.run
echo "CUDA 11.8 toolkit installed at /usr/local/cuda-11.8."

echo "=== [5/6] .bashrc CUDA paths ==="
if ! grep -q 'cuda-11.8/bin' "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" << 'BASHRC_BLOCK'

# --- CUDA 11.8 ---
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
BASHRC_BLOCK
    echo "CUDA paths added to .bashrc."
else
    echo "CUDA paths already present in .bashrc."
fi
export PATH="/usr/local/cuda-11.8/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== [6/6] Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda3_installer.sh
    bash /tmp/miniconda3_installer.sh -b -p "$HOME/miniconda3"
    rm -f /tmp/miniconda3_installer.sh
    echo "Miniconda installed."
else
    echo "Miniconda already present."
fi
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda init bash 2>/dev/null
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo ""
echo "System setup complete. Reboot to activate nvidia-driver-535."
