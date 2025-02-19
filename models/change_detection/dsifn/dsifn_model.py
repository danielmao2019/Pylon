from typing import Tuple, Dict
import torch
from models.change_detection.dsifn.channel_attention import ChannelAttention
from models.change_detection.dsifn.spatial_attention import SpatialAttention
from models.change_detection.dsifn.utils import conv2d_bn


class DSIFN(torch.nn.Module):

    def __init__(self, feature_extractor: torch.nn.Module) -> None:
        super(DSIFN, self).__init__()
        self.feature_extractor = feature_extractor
        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        
        self.sigmoid = torch.nn.Sigmoid()

        # branch1
        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = torch.nn.BatchNorm2d(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = torch.nn.BatchNorm2d(512)
        self.o1_conv3 = torch.nn.Conv2d(512, 1, 1)
        self.trans_conv1 = torch.nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = torch.nn.BatchNorm2d(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = torch.nn.BatchNorm2d(256)
        self.o2_conv4 = torch.nn.Conv2d(256, 1, 1)
        self.trans_conv2 = torch.nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = torch.nn.BatchNorm2d(128)
        self.o3_conv4 = torch.nn.Conv2d(128, 1, 1)
        self.trans_conv3 = torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = torch.nn.BatchNorm2d(64)
        self.o4_conv4 = torch.nn.Conv2d(64, 1, 1)
        self.trans_conv4 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = torch.nn.BatchNorm2d(16)
        self.o5_conv4 = torch.nn.Conv2d(16, 1, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        t1_list = self.feature_extractor(inputs['img_1'])
        t2_list = self.feature_extractor(inputs['img_2'])

        t1_f_l3,t1_f_l8,t1_f_l15,t1_f_l22,t1_f_l29 = t1_list[0],t1_list[1],t1_list[2],t1_list[3],t1_list[4]
        t2_f_l3,t2_f_l8,t2_f_l15,t2_f_l22,t2_f_l29,= t2_list[0],t2_list[1],t2_list[2],t2_list[3],t2_list[4]

        x = torch.cat((t1_f_l29,t2_f_l29),dim=1)
        #optional to use channel attention module in the first combined feature
        # x = self.ca1(x) * x
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.sigmoid(self.o1_conv3(x))

        x = self.trans_conv1(x)
        x = torch.cat((x,t1_f_l22,t2_f_l22),dim=1)
        x = self.ca2(x)*x
        #According to the amount of the training data, appropriately reduce the use of conv layers to prevent overfitting
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        branch_2_out = self.sigmoid(self.o2_conv4(x))

        x = self.trans_conv2(x)
        x = torch.cat((x,t1_f_l15,t2_f_l15),dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        branch_3_out = self.sigmoid(self.o3_conv4(x))

        x = self.trans_conv3(x)
        x = torch.cat((x,t1_f_l8,t2_f_l8),dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        branch_4_out = self.sigmoid(self.o4_conv4(x))

        x = self.trans_conv4(x)
        x = torch.cat((x,t1_f_l3,t2_f_l3),dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        branch_5_out = self.sigmoid(self.o5_conv4(x))

        return branch_5_out,branch_4_out,branch_3_out,branch_2_out,branch_1_out
