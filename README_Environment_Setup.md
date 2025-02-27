# Environment Setup

```bash
conda update -n base -c defaults conda -y
conda create --name Pylon python=3.10 -y
conda activate Pylon
pip install --upgrade pip
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine mmcv==2.0.0 mmdet==3.0.0
pip install mmsegmentation==1.2.2
conda install timm einops scipy scikit-learn scikit-image pycocotools opencv rasterio pytest matplotlib imageio dash tqdm -c conda-forge -y
pip install fvcore triton jsbeautifier
```
