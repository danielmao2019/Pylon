class YTYAttention(torch.nn.Module):

    def __init__(self, channel=512, reduction=16, im_channel=49):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(im_channel, 1, bias=False)
        self.fc2 = torch.nn.Linear(im_channel, 1, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
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
        # b, c, _, _ = x.size()
        img = torch.cat([im1, im2], dim=2)
        origin = img
        im1 = im1.transpose(1, 2)
        im2 = im2.transpose(1, 2)
        im1 = self.fc1(im1)  # 1,512,1
        im2 = self.fc2(im2)  # 1,512,1

        im = torch.cat([im1, im2], dim=2)  # 1,512,2
        im = torch.transpose(im, 1, 2)  # 1,2,512
        im = self.fc(im)  # 1,2,512
        im1 = im[:, 0, :].unsqueeze(1)
        im2 = im[:, 1, :].unsqueeze(1)
        im = torch.cat([im1, im2], dim=2)
        im = im.transpose(1, 2)
        im = self.sigmoid(im)
        img = img.transpose(1, 2)
        res = img * im.expand_as(img)
        res = res.transpose(1, 2)
        return res
