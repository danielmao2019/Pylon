from typing import Optional
import torch
from .binary_dice_loss import BinaryDiceLoss

class DiceLoss(torch.nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        y_predict: A tensor of shape [N, C, *]
        y_true: A tensor of same shape with y_predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    reference:
    * https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
    """
    def __init__(self, weight=None, ignore_index=None, smooth=1.0, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_predict.shape == y_true.shape, 'y_predict & y_true shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        y_predict = torch.nn.functional.softmax(y_predict, dim=1)

        for i in range(y_true.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(y_predict[:, i], y_true[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == y_true.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(y_true.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/y_true.shape[1]