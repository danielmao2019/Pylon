import torch

class DRFD(torch.nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = torch.nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = torch.nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=2, padding=1, groups=dim*2)
        self.act_c = act_layer()
        self.norm_c = norm_layer(dim*2)
        self.max_m = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = norm_layer(dim*2)
        self.fusion = torch.nn.Conv2d(dim*4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        max = self.norm_m(self.max_m(x))                  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))    # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)                # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)                                      # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x
