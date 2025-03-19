from typing import Any, Dict, List, Tuple, Union
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.change_detection.cdmaskformer.set_criterion import SetCriterion
from criteria.vision_2d.change_detection.cdmaskformer.point_features import point_sample, get_uncertain_point_coords_with_randomness
from utils.matcher import HungarianMatcher


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))
    return loss / hw


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))
    return loss / hw


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
        cost_functions = {
            "class": (batch_sigmoid_ce_loss, self.class_weight),
            "mask": (batch_sigmoid_focal_loss, self.mask_weight),
            "dice": (batch_dice_loss, self.dice_weight),
        }
        matcher = HungarianMatcher(
            cost_functions=cost_functions,
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