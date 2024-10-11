from typing import Dict, Optional
import torch


class MultiTaskBaseModel(torch.nn.Module):

    def __init__(
        self,
        backbone: torch.nn.Module,
        decoders: torch.nn.ModuleDict,
        use_attention: Optional[bool] = False,
        channels: Optional[int] = None,
    ) -> None:
        super(MultiTaskBaseModel, self).__init__()
        assert isinstance(backbone, torch.nn.Module)
        assert isinstance(decoders, torch.nn.ModuleDict)
        self.backbone = backbone
        self.decoders = decoders
        if use_attention:
            assert channels is not None
            self.attention_modules = torch.nn.ModuleDict({
                name: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channels, out_channels=channels,
                        kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                    ),
                    torch.nn.Conv2d(
                        in_channels=channels, out_channels=channels,
                        kernel_size=3, stride=1, padding=1, dilation=1, groups=channels, bias=True,
                    ),
                    torch.nn.Conv2d(
                        in_channels=channels, out_channels=channels,
                        kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True,
                    )
                ) for name in decoders
            })
        else:
            self.attention_modules = None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs['image']
        h, w = x.shape[2], x.shape[3]
        shared_rep: torch.Tensor = self.backbone(x)
        outputs: Dict[str, torch.Tensor] = {}
        if self.attention_modules is not None:
            attention_masks: Dict[str, torch.Tensor] = {}
        for name in self.decoders:
            if self.attention_modules is not None:
                mask = self.attention_modules[name](shared_rep)
                assert mask.shape == shared_rep.shape
                task_output = mask * shared_rep
                task_output = self.decoders[name](task_output)
                attention_masks[name] = mask.detach().clone()
            else:
                task_output = self.decoders[name](shared_rep)
            outputs[name] = torch.nn.functional.upsample(task_output, size=(h, w), mode='bilinear')
        outputs['shared_rep'] = shared_rep
        if self.attention_modules:
            outputs['attention_masks'] = attention_masks
        return outputs
