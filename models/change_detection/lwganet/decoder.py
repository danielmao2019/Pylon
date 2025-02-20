import torch

from .supervised_attention import SupervisedAttentionModule

class Decoder(torch.nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_p3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.conv_p2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.mid_d),
            torch.nn.ReLU(inplace=True)
        )
        self.cls = torch.nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + torch.nn.functional.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + torch.nn.functional.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + torch.nn.functional.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5