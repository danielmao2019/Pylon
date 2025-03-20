import torch 
import torch.nn as nn
from models.change_detection.cdmaskformer.backbone.seaformer import SeaFormer_L


class CDMaskFormerBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SeaFormer_L(pretrained=True)

    def forward(self, xA, xB) -> List[torch.Tensor]:
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]
