import numpy as np
import itertools
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from criteria.wrappers import SingleTaskCriterion
from rscd.losses.loss_util.criterion import SetCriterion
from rscd.losses.loss_util.matcher import HungarianMatcher


class CDMaskFormerCriterion(SingleTaskCriterion):
    """Criterion for CDMaskFormer model.
    
    This criterion combines:
    1. Cross-entropy loss for classification
    2. Dice loss for mask prediction
    3. Mask loss for pixel-wise prediction
    
    The losses are weighted and combined with Hungarian matching to assign predictions to ground truth.
    """

    def __init__(
        self,
        class_weight: float = 2.0,
        dice_weight: float = 5.0,
        mask_weight: float = 5.0,
        no_object_weight: float = 0.1,
        dec_layers: int = 10,
        num_classes: int = 1,
        device: str = "cuda:0"
    ) -> None:
        """Initialize the criterion.
        
        Args:
            class_weight: Weight for classification loss
            dice_weight: Weight for dice loss
            mask_weight: Weight for mask loss
            no_object_weight: Weight for no-object class
            dec_layers: Number of decoder layers
            num_classes: Number of classes (including background)
            device: Device to use for computation
        """
        super(CDMaskFormerCriterion, self).__init__()
        self.device = device
        self.class_weight = class_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.no_object_weight = no_object_weight
        self.dec_layers = dec_layers
        self.num_classes = num_classes

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the total loss.
        
        Args:
            y_pred: Dictionary containing model predictions with keys:
                - pred_logits: Classification logits [B, num_queries, num_classes]
                - pred_masks: Mask predictions [B, num_queries, H, W]
                - aux_outputs: List of auxiliary outputs from decoder layers
            y_true: Dictionary containing ground truth with keys:
                - change_map: Binary change map [B, H, W]
                
        Returns:
            Total loss tensor
        """
        # Input validation
        assert isinstance(y_pred, dict)
        assert isinstance(y_true, dict)
        assert set(y_pred.keys()) == {'pred_logits', 'pred_masks', 'aux_outputs'}
        assert set(y_true.keys()) == {'change_map'}

        # Building criterion
        matcher = HungarianMatcher(
            cost_class=self.class_weight,
            cost_mask=self.mask_weight,
            cost_dice=self.dice_weight,
            num_points=12544,
        )

        weight_dict = {"loss_ce": self.class_weight, "loss_mask": self.mask_weight, "loss_dice": self.dice_weight}
        aux_weight_dict = {}
        for i in range(self.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=self.no_object_weight,
            losses=losses,
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            device=torch.device(self.device)
        )

        # Resize predictions to match ground truth
        y_pred["pred_masks"] = F.interpolate(
            y_pred["pred_masks"],
            size=y_true['change_map'].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Resize auxiliary outputs
        for v in y_pred['aux_outputs']:
            v['pred_masks'] = F.interpolate(
                v["pred_masks"],
                size=y_true['change_map'].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Compute losses
        losses = criterion(y_pred, y_true)
        weight_dict = criterion.weight_dict
                    
        loss_ce = 0.0
        loss_dice = 0.0
        loss_mask = 0.0
        for k in list(losses.keys()):
            if k in weight_dict:
                losses[k] *= criterion.weight_dict[k]
                if '_ce' in k:
                    loss_ce += losses[k]
                elif '_dice' in k:
                    loss_dice += losses[k]
                elif '_mask' in k:
                    loss_mask += losses[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        
        loss = loss_ce + loss_dice + loss_mask
        assert loss.ndim == 0, f"{loss.shape=}"
        self.buffer.append(loss.detach().cpu())
        return loss