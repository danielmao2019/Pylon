import torch
import torch.nn.functional as F
from typing import Dict, List
from criteria.wrappers import SingleTaskCriterion
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


def batch_sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2):
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


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, alpha: float = 0.25, gamma: float = 2):
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
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def compute_class_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute classification cost for Hungarian matching.
    """
    out_prob = outputs["pred_logits"].softmax(-1)  # [num_queries, num_classes+1]
    tgt_ids = targets["labels"]
    cost_class = -out_prob[:, tgt_ids]
    return cost_class


def compute_mask_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute mask BCE cost for Hungarian matching using focal loss.
    """
    out_mask = outputs["pred_masks"].flatten(1)  # [num_queries, H*W]
    tgt_mask = targets["masks"].flatten(1)  # [num_targets, H*W]
    cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
    return cost_mask


def compute_dice_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute dice cost for Hungarian matching.
    """
    out_mask = outputs["pred_masks"].flatten(1)  # [num_queries, H*W]
    tgt_mask = targets["masks"].flatten(1)  # [num_targets, H*W]
    cost_dice = batch_dice_loss(out_mask, tgt_mask)
    return cost_dice


class CDMaskFormerCriterion(SingleTaskCriterion):
    """
    Criterion for CDMaskFormer that combines mask classification and mask prediction losses.
    This criterion follows the Mask2Former approach with Hungarian matching.
    
    Class mapping:
    - 0: no object (queries that don't match any target)
    - 1: change class (foreground)
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        class_weight: float = 2.0,
        dice_weight: float = 5.0,
        mask_weight: float = 5.0,
        no_object_weight: float = 0.1,
        dec_layers: int = 10,
    ) -> None:
        """
        Initialize the CDMaskFormer criterion.
        
        Args:
            num_classes: Number of classes for semantic segmentation (default is 1 for binary change detection)
            class_weight: Weight for the classification loss component
            dice_weight: Weight for the dice loss component
            mask_weight: Weight for the mask/BCE loss component
            no_object_weight: Weight for the no-object class
            dec_layers: Number of decoder layers (used for auxiliary losses)
        """
        super(CDMaskFormerCriterion, self).__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.no_object_weight = no_object_weight
        self.dec_layers = dec_layers
        
        # Set up matcher and cost functions
        cost_functions = {
            "cost_class": (compute_class_cost, self.class_weight),
            "cost_mask": (compute_mask_cost, self.mask_weight),
            "cost_dice": (compute_dice_cost, self.dice_weight)
        }
        
        self.matcher = HungarianMatcher(cost_functions=cost_functions)
        
        # Create weight dict for losses
        self.weight_dict = {"loss_ce": self.class_weight, "loss_mask": self.mask_weight, "loss_dice": self.dice_weight}
        aux_weight_dict = {}
        for i in range(self.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
        self.weight_dict.update(aux_weight_dict)
        
        # Create class weights with higher weight for no-object class
        self.register_buffer(
            "empty_weight", 
            torch.ones(self.num_classes + 1)
        )
        self.empty_weight[0] = self.no_object_weight
        
    def _compute_loss(self, y_pred: Dict[str, torch.Tensor], y_true: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute the CDMaskFormer loss.
        
        Args:
            y_pred: Dictionary containing model outputs with keys 'pred_logits', 'pred_masks', etc.
            y_true: List of target dictionaries, one per batch element
            
        Returns:
            Scalar loss tensor
        """
        # Ensure masks are at the right scale
        y_pred["pred_masks"] = F.interpolate(
            y_pred["pred_masks"],
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        
        # Also scale masks in auxiliary outputs if present
        if "aux_outputs" in y_pred:
            for aux_outputs in y_pred["aux_outputs"]:
                aux_outputs["pred_masks"] = F.interpolate(
                    aux_outputs["pred_masks"],
                    scale_factor=(4, 4),
                    mode="bilinear",
                    align_corners=False,
                )
                
        # Find matches between queries and targets
        indices = self.matcher(y_pred, y_true)
        
        # Compute the number of masks for normalization
        num_masks = sum(len(t["labels"]) for t in y_true)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=y_pred["pred_logits"].device)
        
        # Compute classification loss
        loss_ce = self._compute_classification_loss(y_pred, y_true, indices)
        
        # Compute mask losses
        loss_mask, loss_dice = self._compute_mask_losses(y_pred, y_true, indices, num_masks)
        
        # Combine losses
        losses = {
            "loss_ce": loss_ce * self.weight_dict["loss_ce"],
            "loss_mask": loss_mask * self.weight_dict["loss_mask"],
            "loss_dice": loss_dice * self.weight_dict["loss_dice"]
        }
        
        # Compute auxiliary losses if present
        if "aux_outputs" in y_pred:
            for i, aux_outputs in enumerate(y_pred["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, y_true)
                aux_loss_ce = self._compute_classification_loss(aux_outputs, y_true, aux_indices)
                aux_loss_mask, aux_loss_dice = self._compute_mask_losses(aux_outputs, y_true, aux_indices, num_masks)
                
                losses[f"loss_ce_{i}"] = aux_loss_ce * self.weight_dict[f"loss_ce_{i}"]
                losses[f"loss_mask_{i}"] = aux_loss_mask * self.weight_dict[f"loss_mask_{i}"]
                losses[f"loss_dice_{i}"] = aux_loss_dice * self.weight_dict[f"loss_dice_{i}"]
        
        # Compute total loss
        total_loss = sum(losses.values())
        return total_loss
    
    def _compute_classification_loss(self, outputs, targets, indices):
        """Compute the classification loss."""
        pred_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(pred_logits.shape[:2], 0, 
                                   dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce
    
    def _compute_mask_losses(self, outputs, targets, indices, num_masks):
        """Compute the mask focal loss and dice loss."""
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        
        masks = [t["masks"] for t in targets]
        target_masks = torch.cat([t[i] for t, (_, i) in zip(masks, indices)], dim=0)
        target_masks = target_masks.to(src_masks)
        
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        
        loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_masks)
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        
        return loss_mask, loss_dice
    
    def _get_src_permutation_idx(self, indices):
        """Get source permutation indices for the loss computation."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Get target permutation indices for the loss computation."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
