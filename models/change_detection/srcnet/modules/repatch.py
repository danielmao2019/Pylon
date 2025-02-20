import torch


class Repatch(torch.nn.Module):

    def __init__(self, in_ch, patch_size):
        super().__init__()
        self.patchup = torch.nn.ConvTranspose2d(
            in_ch, 2, kernel_size=patch_size, stride=patch_size
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, a, b):
        x1, x2 = self.patchup(a), self.patchup(b)
        x1P, x2P = self.softmax(x1), self.softmax(x2)
        spA = torch.zeros_like(x1P)
        spA[:, 1, :, :] = (
            x1P[:, 1, :, :] * x2P[:, 0, :, :] + x1P[:, 0, :, :] * x2P[:, 1, :, :]
        )
        spA[:, 0, :, :] = (
            x1P[:, 0, :, :] * x2P[:, 0, :, :] + x1P[:, 1, :, :] * x2P[:, 1, :, :]
        )
        return spA
