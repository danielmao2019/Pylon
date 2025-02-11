import torch


class DWConv(torch.nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = torch.nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
