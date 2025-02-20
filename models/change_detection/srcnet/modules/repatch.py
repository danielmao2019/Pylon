import torch


class Repatch(torch.nn.Module):

    def __init__(self, in_ch, patch_size):
        super(Repatch, self).__init__()
        self.patchup = torch.nn.ConvTranspose2d(
            in_ch, 2, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, a, b):
        x1, x2 = self.patchup(a), self.patchup(b)
        spA = torch.zeros_like(x1)
        spA[:, 1, :, :] = (
            x1[:, 1, :, :] * x2[:, 0, :, :] + x1[:, 0, :, :] * x2[:, 1, :, :]
        )
        spA[:, 0, :, :] = (
            x1[:, 0, :, :] * x2[:, 0, :, :] + x1[:, 1, :, :] * x2[:, 1, :, :]
        )
        return spA
