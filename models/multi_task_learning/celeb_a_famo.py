from typing import Dict
import torch


class CelebA_FAMO(torch.nn.Module):
    __doc__ = r"""
    FAMO model for the CelebA dataset multi-task attribute classification.
    The model uses the Fast Adaptive Multitask Optimization (FAMO) approach for training.
    
    Reference: https://github.com/Cranial-XIX/FAMO/blob/main/experiments/celeba/models.py
    
    For multi-task learning documentation, see: docs/datasets/multi_task/celeb_a.md
    """

    LABEL_NAMES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                   'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    @staticmethod
    def _conv_unit_(in_channels: int, out_channels: int):
        return [
            torch.nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        ]

    @staticmethod
    def _linear_unit_(in_features: int, out_features: int):
        return [
            torch.nn.Linear(in_features, out_features, bias=False),
            torch.nn.BatchNorm1d(out_features),
            torch.nn.ReLU(inplace=True),
        ]

    BASE_CHANNELS = 64

    def __init__(self, width_multiplier: float = 1.0):
        super(CelebA_FAMO, self).__init__()
        # input checks
        assert type(width_multiplier) == float, f"{type(width_multiplier)=}"
        assert width_multiplier > 0
        channels_multiple = int(self.BASE_CHANNELS * width_multiplier)
        assert channels_multiple > 0
        # define backbone
        self.backbone = torch.nn.Sequential(
            *self._conv_unit_(3, channels_multiple),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            *self._conv_unit_(1*channels_multiple, 2*channels_multiple),
            *self._conv_unit_(2*channels_multiple, 2*channels_multiple),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            *self._conv_unit_(2*channels_multiple, 4*channels_multiple),
            *self._conv_unit_(4*channels_multiple, 4*channels_multiple),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            *self._conv_unit_(4*channels_multiple, 8*channels_multiple),
            *self._conv_unit_(8*channels_multiple, 8*channels_multiple),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            *self._linear_unit_(8*channels_multiple, 8*channels_multiple),
            *self._linear_unit_(8*channels_multiple, 8*channels_multiple),
        )
        # define prediction heads
        self.pred_heads = torch.nn.ModuleList([torch.nn.Linear(8*channels_multiple, 1) for _ in range(40)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_rep = self.backbone(x)
        y = [self.pred_heads[task](shared_rep) for task in range(40)]
        output = dict(zip(self.LABEL_NAMES, y))
        output['shared_rep'] = shared_rep
        return output

    def shared_parameters(self):
        return (p for p in self.backbone.parameters())

    def task_specific_parameters(self):
        return_list = []
        for task in range(40):
            return_list += [p for p in self.pred_heads[task].parameters()]
        return return_list

    def last_shared_parameters(self):
        return []
