import torch
from models.change_detection.srcnet.modules.patch_emb import PatchEmb
from models.change_detection.srcnet.modules.festage import FEStage
from models.change_detection.srcnet.modules.repatch import Repatch
from models.change_detection.srcnet.modules.pmffm import PMFFM
from models.change_detection.srcnet.modules.src_block import SRCBlock


class SRCNet(torch.nn.Module):

    def __init__(self):
        super(SRCNet, self).__init__()
        patch_size = 8
        dim = 256
        n1 = n2 = 4
        # Patch Embedding
        self.PE = PatchEmb(dim=dim, patch_size=patch_size)
        # Feature Extraction
        self.FES = FEStage(dim, n1)
        # Feature Fusion
        self.repatch = Repatch(in_ch=dim, patch_size=patch_size)
        self.mix = PMFFM(dim=dim, k=16, m=8)
        # Change Prediction
        self.CPBlocks = torch.nn.ModuleList([SRCBlock(dim) for _ in range(n2)])
        # Patch Combining
        self.patchup = torch.nn.ConvTranspose2d(
            dim, 32, kernel_size=patch_size, stride=patch_size
        )
        self.norm = torch.nn.BatchNorm2d(32)
        self.gelu = torch.nn.GELU()
        self.final = torch.nn.Conv2d(32, 2, kernel_size=1)
        # Loss
        self.sigma = torch.nn.Parameter(torch.ones(3))

    def forward(self, a, b):
        if self.training:
            a, b = self.randomAB(a, b)
        # Patch Embedding
        x1, x2 = self.PE(a), self.PE(b)
        # Feature Extraction
        x1, x2, diff = self.FES(x1, x2)
        # Feature Fusion
        Dis = self.repatch(x1, x2)
        x = self.mix(x1, x2)
        # Change Prediction
        for blk in self.CPBlocks:
            x = blk(x)
        # Patch Combining
        out = self.patchup(x)
        out = self.gelu(self.norm(out))
        out = self.final(out)
        return (out,), (Dis,), (diff,), self.sigma

    def randomAB(self, a, b):
        shape = (a.shape[0], 1, 1, 1)
        random_tensor = torch.rand(shape, dtype=a.dtype, device=a.device)
        random_tensor = random_tensor + 0.5
        random_tensor = random_tensor.floor_()
        return a * random_tensor + b * (1 - random_tensor), b * random_tensor + a * (
            1 - random_tensor
        )
