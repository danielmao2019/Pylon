import torch
from models.change_detection.srcnet.modules.layer_norm import LayerNorm
from models.change_detection.srcnet.modules.drop_path import DropPath
from models.change_detection.srcnet.modules.grn import GRN


class SRCBlock(torch.nn.Module):

    def __init__(self, dim, drop_rate=0.2, mode="LN"):
        super().__init__()

        self.depthconv1 = torch.nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.depthconv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.depthconv3 = torch.nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        if mode == "BN":
            self.norm = torch.nn.BatchNorm2d(dim)
        else:
            self.norm = LayerNorm(dim, data_format="channels_first")

        self.pointconv1 = torch.nn.Linear(dim, 4 * dim)
        self.gelu = torch.nn.GELU()
        self.grn = GRN(4 * dim)

        self.pointconv2 = torch.nn.Linear(4 * dim, dim)

        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else torch.nn.Identity()

    def forward(self, x):
        shortcut = x

        x = (self.depthconv1(x) + self.depthconv2(x) + self.depthconv3(x)) / 3
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.pointconv1(x)
        x = self.gelu(x)
        x = self.grn(x)
        x = self.pointconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = shortcut + self.drop_path(x)
        return x
