import torch
import torch.nn.functional as F
from models.change_detection.change_former.modules.conv_layer import ConvLayer
from models.change_detection.change_former.modules.upsample_conv_layer import UpsampleConvLayer
from models.change_detection.change_former.modules.residual_block import ResidualBlock


class convprojection(torch.nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection,self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = torch.nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = torch.nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = torch.nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = torch.nn.Sequential( ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 2, kernel_size=3, stride=1, padding=1)

        self.active = torch.nn.Tanh()

    def forward(self,x1,x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,-1,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)

        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0,-1,0,0)
            res32x = F.pad(res32x,p2d,"constant",0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0,0,0,-1)
            res32x = F.pad(res32x,p2d,"constant",0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x)

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,-1,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0,-1,0,0)
            res16x = F.pad(res16x,p2d,"constant",0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0,0,0,-1)
            res16x = F.pad(res16x,p2d,"constant",0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x
