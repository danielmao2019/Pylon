import torch

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = torch.nn.Sequential(
            torch.nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.out_d),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = torch.nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out