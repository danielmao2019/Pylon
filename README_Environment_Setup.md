# Environment Setup

```bash
conda create --name Pylon python=3.10 -y
conda activate Pylon
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install scipy pycocotools opencv pytest matplotlib -c conda-forge -y
pip install jsbeautifier wandb
```
