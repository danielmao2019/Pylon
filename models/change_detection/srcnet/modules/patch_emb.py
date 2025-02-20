import torch


class PatchEmb(torch.nn.Module):

    def __init__(self, dim, patch_size):
        super().__init__()
        self.GenPatch = torch.nn.Conv2d(3, dim // 4, kernel_size=4, stride=4)
        self.ln = torch.nn.BatchNorm2d(dim // 4)
        self.GenPatch2 = torch.nn.Conv2d(
            dim // 4, dim, kernel_size=patch_size // 4, stride=patch_size // 4
        )

    def forward(self, x):
        x = self.GenPatch(x)
        x = self.ln(x)
        x = self.GenPatch2(x)
        return x
