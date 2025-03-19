import torch.nn as nn
from models.change_detection.cdmaskformer.backbones.seaformer import SeaFormer_L

class Base(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'Seaformer':
            self.backbone = SeaFormer_L(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, xA, xB):
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]
