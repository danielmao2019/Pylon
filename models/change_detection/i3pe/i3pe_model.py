import torch.nn as nn
from models.change_detection.i3pe.resnet_18_34 import ResNet18
from models.change_detection.i3pe.resnet_50_101 import ResNet50
import torch
import torch.nn.functional as F


class I3PEModel(nn.Module):

    def __init__(self, pretrained=True, output_stride=16, BatchNorm=nn.BatchNorm2d, Backbone='ResNet50'):
        super(I3PEModel, self).__init__()
        if Backbone == 'ResNet50':
            self.encoder = ResNet50(BatchNorm=BatchNorm, pretrained=pretrained, output_stride=output_stride)

            self.fuse_layer_4 = nn.Conv2d(kernel_size=1, in_channels=4096, out_channels=128)
            self.fuse_layer_3 = nn.Conv2d(kernel_size=1, in_channels=2048, out_channels=128)
            self.fuse_layer_2 = nn.Conv2d(kernel_size=1, in_channels=1024, out_channels=128)
            self.fuse_layer_1 = nn.Conv2d(kernel_size=1, in_channels=512, out_channels=128)

            self.smooth_layer_3 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
            self.smooth_layer_2 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
            self.smooth_layer_1 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)

            self.main_clf_1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        else:
            self.encoder = ResNet18(BatchNorm=BatchNorm, pretrained=pretrained, output_stride=output_stride)

            self.fuse_layer_4 = nn.Conv2d(kernel_size=1, in_channels=1024, out_channels=128)
            self.fuse_layer_3 = nn.Conv2d(kernel_size=1, in_channels=512, out_channels=128)
            self.fuse_layer_2 = nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128)
            self.fuse_layer_1 = nn.Conv2d(kernel_size=1, in_channels=128, out_channels=128)

            self.smooth_layer_3 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
            self.smooth_layer_2 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)
            self.smooth_layer_1 = nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1)

            self.main_clf_1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, img_1: torch.Tensor, img_2: torch.Tensor) -> torch.Tensor:
        _, pre_low_level_feat_1, pre_low_level_feat_2, pre_low_level_feat_3, pre_output = self.encoder(img_1)
        _, post_low_level_feat_1, post_low_level_feat_2, post_low_level_feat_3, post_output = self.encoder(img_2)

        p4 = torch.cat([pre_output, post_output], dim=1)
        p4 = self.fuse_layer_4(p4)

        p3 = torch.cat([pre_low_level_feat_3, post_low_level_feat_3], dim=1)
        p3 = self.fuse_layer_3(p3)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        p2 = torch.cat([pre_low_level_feat_2, post_low_level_feat_2], dim=1)
        p2 = self.fuse_layer_2(p2)
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        p1 = torch.cat([pre_low_level_feat_1, post_low_level_feat_1], dim=1)
        p1 = self.fuse_layer_1(p1)
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        output_1 = self.main_clf_1(p1)
        output_1 = F.interpolate(output_1, size=img_1.size()[-2:], mode='bilinear')
        return output_1
