import torch


class TYAttention(torch.nn.Module):
    def __init__(self, in_channel=1024, im_channel=49, gamma=2, b=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(im_channel, 1, bias=False)
        self.fc2 = torch.nn.Linear(im_channel, 1, bias=False)
        self.SiLU = torch.nn.SiLU()
        self.bn = torch.nn.BatchNorm1d(2)
        self.gamma = gamma
        self.b = b
        self.sigmoid = torch.nn.Sigmoid()
        self.in_channel = in_channel
        t = int(abs((math.log(self.in_channel, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = torch.nn.Conv1d(2, 2, kernel_size=k, padding=int(k / 2), bias=False)
        self.conv2 = torch.nn.Conv1d(2, 2, kernel_size=k, padding=int(k / 2), bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, im1, im2):
        img = torch.cat([im1, im1], dim=2)  # B,49,2048
        img = img.transpose(1, 2)
        origin = img
        im1 = im1.transpose(-1, -2)  # B,1024,49
        im2 = im2.transpose(-1, -2)

        im1 = self.fc1(im1)  # B,1024,1
        im2 = self.fc2(im2)
        im = torch.cat([im1, im2], dim=2)  # B,1024,2
        im = self.conv(im.transpose(-1, -2))  # B,2,1024
        im = self.SiLU(im)
        # im = self.bn(im)
        im = self.conv2(im)
        im1 = im[:, 0, :].unsqueeze(1)
        im2 = im[:, 1, :].unsqueeze(1)
        im = torch.cat([im1, im2], dim=2)  # B,1,2048
        im = im.transpose(1, 2)
        im = self.sigmoid(im)
        res = img * im.expand_as(img) + origin
        return res.transpose(1, 2)
