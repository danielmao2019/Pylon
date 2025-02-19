import torch

class GA12(torch.nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = torch.nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = torch.nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = torch.nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = torch.nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = torch.nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = torch.nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = torch.nn.Conv2d(2, 2, 7, padding=3)
        self.conv = torch.nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = torch.nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x
