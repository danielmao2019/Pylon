import torch

class LA(torch.nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x