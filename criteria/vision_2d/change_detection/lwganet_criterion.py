from typing import Dict
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion


class LWGANetCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(LWGANetCriterion, self).__init__()
        self.ce_criterion = criteria.vision_2d.SemanticSegmentationCriterion()
        self.dice_criterion = criteria.vision_2d.DiceLoss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_true.keys()) == {'change_map'}

        # Compute the binary cross entropy loss for each mask and sum them up
        ce_loss = sum( self.ce_criterion(mask, y_true['change_map'])
            for mask in y_pred.values())
    
        # Compute the dice loss for each mask and sum them up
        dice_loss = sum(self.dice_criterion(y_pred=mask, y_true=y_true['change_map'])
            for mask in y_pred.values())
        
        total_loss = ce_loss + dice_loss
        
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        
        return total_loss