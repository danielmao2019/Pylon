from typing import Dict, Optional
import torch


class MultiTaskBaseModel(torch.nn.Module):
    """
    Base class for multi-task learning models with shared backbone and task-specific decoders.
    
    This flexible architecture allows for various multi-task learning approaches by combining
    a shared feature extractor (backbone) with task-specific decoders. It optionally supports
    attention mechanisms for task-specific feature refinement.
    
    For multi-task learning documentation, see the dataset-specific documentation files:
    - docs/datasets/multi_task/celeb_a.md
    - docs/datasets/multi_task/city_scapes.md
    - docs/datasets/multi_task/multi_task_facial_landmark.md
    - docs/datasets/multi_task/nyu_v2.md
    - docs/datasets/multi_task/pascal_context.md
    - docs/datasets/multi_task/multi_mnist.md
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        decoders: torch.nn.ModuleDict,
        return_shared_rep: Optional[bool] = True,
        use_attention: Optional[bool] = False,
        attn_in: Optional[int] = None,
    ) -> None:
        r"""
        Args:
            return_shared_rep (bool): for single-task learning compatibility.
            attn_in (int): Only used when use_attention is True.
        """
        super(MultiTaskBaseModel, self).__init__()
        assert isinstance(backbone, torch.nn.Module)
        assert isinstance(decoders, torch.nn.ModuleDict)
        self.backbone = backbone
        self.decoders = decoders
        self.return_shared_rep = return_shared_rep
        if use_attention:
            assert attn_in is not None
            self.attention_modules = torch.nn.ModuleDict({
                name: torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=attn_in, out_channels=attn_in,
                        kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                    ),
                    torch.nn.Conv2d(
                        in_channels=attn_in, out_channels=attn_in,
                        kernel_size=3, stride=1, padding=1, dilation=1, groups=attn_in, bias=True,
                    ),
                    torch.nn.Conv2d(
                        in_channels=attn_in, out_channels=attn_in,
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
        if self.return_shared_rep:
            outputs['shared_rep'] = shared_rep
        if self.attention_modules:
            outputs['attention_masks'] = attention_masks
        return outputs
