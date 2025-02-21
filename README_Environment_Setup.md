# Environment Setup

```bash
conda update -n base -c defaults conda -y
conda create --name Pylon python=3.10 -y
conda activate Pylon
pip install --upgrade pip
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install timm einops scipy scikit-learn scikit-image pycocotools opencv rasterio pytest matplotlib imageio dash tqdm -c conda-forge -y
pip install fvcore triton jsbeautifier
```
