import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import models


class Resnet50WithFPN(torch.nn.Module):

    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        m = resnet50()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return list(x.values())[0]


model_config = {
    'class': models.change_detection.ChangeStar,
    'args': {
        'encoder': Resnet50WithFPN(),
        'change_decoder_cfg': {
            'in_channels': 256*2,
            'mid_channels': 16,
            'out_channels': 2,
            'drop_rate': 0.2,
            'scale_factor': 4.0,
            'num_convs': 4,
        },
        'semantic_decoder_cfg': {
            'in_channels': 256,
            'out_channels': 5,
            'scale_factor': 4.0,
        },
    },
}
