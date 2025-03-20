from typing import List
import torch 
from models.change_detection.cdmaskformer.seaformer import SeaFormer_L


class CDMaskFormerBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SeaFormer_L(pretrained=True)

    def forward(self, xA, xB) -> List[torch.Tensor]:
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)

        return [featuresA, featuresB]
