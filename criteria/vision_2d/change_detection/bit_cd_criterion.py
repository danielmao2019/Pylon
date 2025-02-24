from typing import Dict
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion


class BitCdCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(BitCdCriterion, self).__init__()
        # unbounded
        self.ce_criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true: Dict[str, torch.Tensor], reduction='mean', ignore_index=255) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_true.keys()) == {'change_map'}

        # Compute the binary cross entropy loss for each mask and sum them up
        total_loss = self.ce_criterion(y_pred, y_true['change_map'], ignore_index=ignore_index, reduction=reduction)

        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())

        return total_loss