import torch

class SupervisedAttentionModule(torch.nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = torch.nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = torch.nn.Sequential(
            torch.nn.Conv2d(2, self.mid_d, kernel_size=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask
