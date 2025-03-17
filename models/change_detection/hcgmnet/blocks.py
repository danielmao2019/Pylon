import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Padding added to all sides of the input. Default: 0.
        dilation (int, optional): Spacing between kernel elements. Default: 1.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChangeGuideModule(nn.Module):
    """
    Change Guide Module for attention mechanism in change detection.
    
    This module uses a guiding map to enhance feature representation by applying
    attention mechanisms based on the change map.
    
    Args:
        in_dim (int): Number of input channels.
    """
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        # For multi-class output, extract the "change" class logits (class index 1)
        if guiding_map0.size(1) > 1:
            guiding_map0 = guiding_map0[:, 1:2, :, :]  # Shape: [B, 1, H, W]
        
        # Resize guiding map to match feature map dimensions
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        
        # Convert logits to probability
        guiding_map = F.sigmoid(guiding_map0)

        # Apply attention mechanism
        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # Apply residual connection with learnable weight
        out = self.gamma * out + x

        return out
