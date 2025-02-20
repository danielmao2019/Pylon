import torch

class TemporalFeatureFusionModule(torch.nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = torch.nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            torch.nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            torch.nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            torch.nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = torch.nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = torch.nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1, x2):
        # temporal fusion
        x = torch.abs(x1 - x2)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out
