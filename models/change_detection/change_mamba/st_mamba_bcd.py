from typing import Dict
import torch
import torch.nn.functional as F
from models.change_detection.change_mamba.modules.backbone_vssm import Backbone_VSSM
from models.change_detection.change_mamba.modules.vmamba import LayerNorm2d
from models.change_detection.change_mamba.modules.change_decoder import ChangeDecoder


class STMambaBCD(torch.nn.Module):

    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=torch.nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=torch.nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=torch.nn.SiLU,
            gelu=torch.nn.GELU,
            relu=torch.nn.ReLU,
            sigmoid=torch.nn.Sigmoid,
        )

        norm_layer: torch.nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: torch.nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: torch.nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pre_data, post_data = inputs['img_1'], inputs['img_2']
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output
