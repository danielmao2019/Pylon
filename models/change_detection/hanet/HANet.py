from typing import Dict
import torch
import torch.nn as nn
from torch.nn import Softmax
import math


class CAM_Module(nn.Module):
    """Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        # Project to query and key with scaling
        scale = 1.0 / math.sqrt(height * width)
        proj_query = x.view(m_batchsize, C, -1) * math.sqrt(scale)  # Scale before bmm
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) * math.sqrt(scale)  # Scale before bmm

        energy = torch.bmm(proj_query, proj_key)  # Already scaled because inputs are scaled
        max_energy = torch.max(energy, -1, keepdim=True)[0]
        energy_new = max_energy.expand_as(energy) - energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class Conv_CAM_Layer(nn.Module):
    def __init__(self, in_ch, out_in, use_pam=False):
        super(Conv_CAM_Layer, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            CAM_Module(32),
            nn.Conv2d(32, out_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_in),
            nn.PReLU()
        )

    def forward(self, x):
        return self.attn(x)


class FEC(nn.Module):
    """Feature extraction cell"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super(FEC, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class RowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, use_pam=False):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)

        # size = (b*h,w,w) [:,i,j] represents the product of all channel values of Q at position Wi of all h with all channel values of K at position Wj of all h
        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)  # softmax along columns, sum of all columns in a row equals 1

        # size = (b*h,c1,w) transpose row_attn to make sum of all rows in a column equals 1
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)  # size = (b,c1,h,w)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, use_pam=False):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j] represents the product of all channel values of Q at position Hi of all W with all channel values of K at position Hj of all W
        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)  # softmax along columns, sum of all columns in a row equals 1

        # size = (b*w,c1,h) transpose col_attn to make sum of all rows in a column equals 1
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)  # size = (b,c1,h,w)
        out = self.gamma * out + x
        return out


class HANet(nn.Module):
    """HANet"""
    def __init__(self, in_ch=3, ou_ch=2):
        super(HANet, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 40  # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 32, 64, 128, 256, 512

        self.conv0_0 = nn.Conv2d(3, n1, kernel_size=5, padding=2, stride=1)
        self.conv0 = FEC(filters[0], filters[0], filters[0])
        self.conv2 = FEC(filters[0], filters[1], filters[1])
        self.conv4 = FEC(filters[1], filters[2], filters[2])
        self.conv5 = FEC(filters[2], filters[3], filters[3])
        self.conv6 = nn.Conv2d(600, filters[1], kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(filters[1], ou_ch, kernel_size=3, padding=1, bias=False)

        self.conv6_1_1 = nn.Conv2d(filters[0] * 2, filters[0], padding=1, kernel_size=3, groups=filters[0] // 2, dilation=1)
        self.conv6_1_2 = nn.Conv2d(filters[0] * 2, filters[0], padding=2, kernel_size=3, groups=filters[0] // 2, dilation=2)
        self.conv6_1_3 = nn.Conv2d(filters[0] * 2, filters[0], padding=3, kernel_size=3, groups=filters[0] // 2, dilation=3)
        self.conv6_1_4 = nn.Conv2d(filters[0] * 2, filters[0], padding=4, kernel_size=3, groups=filters[0] // 2, dilation=4)
        self.conv1_1 = nn.Conv2d(filters[0] * 4, filters[0], kernel_size=1, stride=1)

        self.conv6_2_1 = nn.Conv2d(filters[1] * 2, filters[1], padding=1, kernel_size=3, groups=filters[1] // 2, dilation=1)
        self.conv6_2_2 = nn.Conv2d(filters[1] * 2, filters[1], padding=2, kernel_size=3, groups=filters[1] // 2, dilation=2)
        self.conv6_2_3 = nn.Conv2d(filters[1] * 2, filters[1], padding=3, kernel_size=3, groups=filters[1] // 2, dilation=3)
        self.conv6_2_4 = nn.Conv2d(filters[1] * 2, filters[1], padding=4, kernel_size=3, groups=filters[1] // 2, dilation=4)
        self.conv2_1 = nn.Conv2d(filters[1] * 4, filters[1], kernel_size=1, stride=1)

        self.conv6_3_1 = nn.Conv2d(filters[2] * 2, filters[2], padding=1, kernel_size=3, groups=filters[2] // 2, dilation=1)
        self.conv6_3_2 = nn.Conv2d(filters[2] * 2, filters[2], padding=2, kernel_size=3, groups=filters[2] // 2, dilation=2)
        self.conv6_3_3 = nn.Conv2d(filters[2] * 2, filters[2], padding=3, kernel_size=3, groups=filters[2] // 2, dilation=3)
        self.conv6_3_4 = nn.Conv2d(filters[2] * 2, filters[2], padding=4, kernel_size=3, groups=filters[2] // 2, dilation=4)
        self.conv3_1 = nn.Conv2d(filters[2] * 4, filters[2], kernel_size=1, stride=1)

        self.conv6_4_1 = nn.Conv2d(filters[3] * 2, filters[3], padding=1, kernel_size=3, groups=filters[3] // 2, dilation=1)
        self.conv6_4_2 = nn.Conv2d(filters[3] * 2, filters[3], padding=2, kernel_size=3, groups=filters[3] // 2, dilation=2)
        self.conv6_4_3 = nn.Conv2d(filters[3] * 2, filters[3], padding=3, kernel_size=3, groups=filters[3] // 2, dilation=3)
        self.conv6_4_4 = nn.Conv2d(filters[3] * 2, filters[3], padding=4, kernel_size=3, groups=filters[3] // 2, dilation=4)
        self.conv4_1 = nn.Conv2d(filters[3] * 4, filters[3], kernel_size=1, stride=1)

        # SA
        self.cam_attention_1 = Conv_CAM_Layer(filters[0], filters[0], False)  # SA4
        self.cam_attention_2 = Conv_CAM_Layer(filters[1], filters[1], False)  # SA3
        self.cam_attention_3 = Conv_CAM_Layer(filters[2], filters[2], False)  # SA2
        self.cam_attention_4 = Conv_CAM_Layer(filters[3], filters[3], False)  # SA1

        # Row Attention
        self.row_attention_1 = RowAttention(filters[0], filters[0], False)  # SA4
        self.row_attention_2 = RowAttention(filters[1], filters[1], False)  # SA3
        self.row_attention_3 = RowAttention(filters[2], filters[2], False)  # SA2
        self.row_attention_4 = RowAttention(filters[3], filters[3], False)  # SA1

        # Col Attention
        self.col_attention_1 = ColAttention(filters[0], filters[0], False)  # SA4
        self.col_attention_2 = ColAttention(filters[1], filters[1], False)  # SA3
        self.col_attention_3 = ColAttention(filters[2], filters[2], False)  # SA2
        self.col_attention_4 = ColAttention(filters[3], filters[3], False)  # SA1

        self.c4_conv = nn.Conv2d(filters[3], filters[1], kernel_size=3, padding=1)
        self.c3_conv = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.c2_conv = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.c1_conv = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.pool1 = nn.AdaptiveAvgPool2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d(32)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.Up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x1 = inputs['img_1']
        x2 = inputs['img_2']
        # The first branch
        x1 = self.conv0(self.conv0_0(x1))  # Output of the first scale
        x3 = self.conv2(self.pool1(x1))
        x4 = self.conv4(self.pool2(x3))
        A_F4 = self.conv5(self.pool3(x4))

        x2 = self.conv0(self.conv0_0(x2))
        x5 = self.conv2(self.pool1(x2))
        x6 = self.conv4(self.pool2(x5))
        A_F8 = self.conv5(self.pool3(x6))

        print('Current module: HANet-WHU-Fixed15-Epo50')
        c4_1 = self.conv4_1(
            torch.cat([self.conv6_4_1(torch.cat([A_F4, A_F8], 1)), self.conv6_4_2(torch.cat([A_F4, A_F8], 1)),
                      self.conv6_4_3(torch.cat([A_F4, A_F8], 1)), self.conv6_4_4(torch.cat([A_F4, A_F8], 1))], 1))
        c4 = self.cam_attention_4(c4_1) + self.row_attention_4(self.col_attention_4(c4_1))

        c3_1 = (self.conv3_1(torch.cat(
            [self.conv6_3_1(torch.cat([x4, x6], 1)), self.conv6_3_2(torch.cat([x4, x6], 1)),
             self.conv6_3_3(torch.cat([x4, x6], 1)), self.conv6_3_4(torch.cat([x4, x6], 1))], 1)))
        c3 = torch.cat([(self.cam_attention_3(c3_1) + self.row_attention_3(self.col_attention_3(c3_1))), self.Up1(c4)], 1)

        c2_1 = (self.conv2_1(torch.cat(
            [self.conv6_2_1(torch.cat([x3, x5], 1)), self.conv6_2_2(torch.cat([x3, x5], 1)),
             self.conv6_2_3(torch.cat([x3, x5], 1)), self.conv6_2_4(torch.cat([x3, x5], 1))], 1)))
        c2 = torch.cat([(self.cam_attention_2(c2_1) + self.row_attention_2(self.col_attention_2(c2_1))), self.Up1(c3)], 1)

        c1_1 = (self.conv1_1(torch.cat(
            [self.conv6_1_1(torch.cat([x1, x2], 1)), self.conv6_1_2(torch.cat([x1, x2], 1)),
             self.conv6_1_3(torch.cat([x1, x2], 1)), self.conv6_1_4(torch.cat([x1, x2], 1))], 1)))
        c1 = torch.cat([(self.cam_attention_1(c1_1) + self.row_attention_1(self.col_attention_1(c1_1))), self.Up1(c2)], 1)

        c1 = self.conv6(c1)
        out1 = self.conv7(c1)

        return out1
