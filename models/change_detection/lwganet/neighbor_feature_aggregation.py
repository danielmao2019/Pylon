import torch

from .feature_fusion import FeatureFusionModule

class NeighborFeatureAggregation(torch.nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(NeighborFeatureAggregation, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 2
        self.conv_scale2_c2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, self.in_d[1], self.out_d)
        # scale 3
        self.conv_scale3_c2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)
        # scale 4
        self.conv_scale4_c3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)
        # scale 5
        self.conv_scale5_c4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_scale5_c5 = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_scale2_c2(c2)

        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = torch.nn.functional.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')

        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)
        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)

        c3_s3 = self.conv_scale3_c3(c3)

        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = torch.nn.functional.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')

        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)
        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)

        c4_s4 = self.conv_scale4_c4(c4)

        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = torch.nn.functional.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')

        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)
        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5