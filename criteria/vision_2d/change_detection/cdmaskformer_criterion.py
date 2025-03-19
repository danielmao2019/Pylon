import torch
import torch.nn.functional as F
from typing import Dict
from criteria.wrappers import SingleTaskCriterion
from utils.matcher import HungarianMatcher


def batch_dice_loss(inputs, targets, eps: float = 1e-8):
    """
    Compute dice loss for a batch of predictions.
    
    Args:
        inputs: [N, L] input logits
        targets: [N, L] ground truth binary masks
        eps: small constant to avoid division by zero
        
    Returns:
        Loss tensor of shape [N]
    """
    # Apply sigmoid to get probabilities
    inputs = inputs.sigmoid()
    
    # Flatten inputs and targets
    numerator = 2 * (inputs * targets).sum(dim=-1)
    denominator = inputs.sum(dim=-1) + targets.sum(dim=-1)
    
    # Compute dice coefficient and loss
    loss = 1 - (numerator + eps) / (denominator + eps)
    
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Compute sigmoid focal loss for a batch of predictions.
    
    Args:
        inputs: [N, L] input logits
        targets: [N, L] ground truth binary masks
        alpha: weighting factor for balancing positive/negative examples
        gamma: focusing parameter for hard example mining
        
    Returns:
        Loss tensor of shape [N, L]
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss


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


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


class CDMaskFormerCriterion(SingleTaskCriterion):
    """
    This class computes the loss for CDMaskFormer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and outputs
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes=1, class_weight=2.0, dice_weight=5.0, 
                 mask_weight=5.0, no_object_weight=0.1, dec_layers=10):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            class_weight: weight for the classification loss
            mask_weight: weight for the mask loss (focal loss)
            dice_weight: weight for the dice loss
            no_object_weight: weight for the no-object class
            dec_layers: number of decoder layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight
        self.dec_layers = dec_layers
        
        # Initialize matcher
        matcher = HungarianMatcher(
            cost_functions={
                "cost_class": (
                    lambda outputs, targets: -outputs["pred_logits"].softmax(-1)[:, targets["labels"]],
                    class_weight
                ),
                "cost_mask": (
                    lambda outputs, targets: batch_sigmoid_focal_loss(outputs["pred_masks"].flatten(1), targets["masks"].flatten(1)),
                    mask_weight
                ),
                "cost_dice": (
                    lambda outputs, targets: batch_dice_loss(outputs["pred_masks"].flatten(1), targets["masks"].flatten(1)),
                    dice_weight
                )
            }
        )
        self.matcher = matcher
        
        # Set up weight dictionary
        self.weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
        self.weight_dict.update(aux_weight_dict)
        
        # Create empty weight for cross-entropy loss
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = no_object_weight
        self.register_buffer('empty_weight', empty_weight)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through CDMaskFormerCriterion.
        
        Args:
            y_pred: Dict with model predictions containing 'pred_logits' and 'pred_masks'
            y_true: Dict containing 'change_map' key with tensor of shape [B, H, W]
            
        Returns:
            Total loss
        """
        # First convert change maps to instance format
        targets = self._get_targets(y_true['change_map'])
        
        # Then upsample the masks to the appropriate size
        outputs = y_pred.copy()
        outputs["pred_masks"] = F.interpolate(
            outputs["pred_masks"],
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        
        # Also upsample any auxiliary outputs
        if "aux_outputs" in outputs:
            for v in outputs['aux_outputs']:
                v['pred_masks'] = F.interpolate(
                    v["pred_masks"],
                    scale_factor=(4, 4),
                    mode="bilinear",
                    align_corners=False,
                )
                
        # Find the best matching between predictions and ground truth
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs["pred_logits"].device)
        num_masks = max(num_masks.item(), 1)  # Ensure num_masks is at least 1
        
        # Compute all loss components
        losses = {}
        losses.update(self._get_losses(outputs, targets, indices, num_masks))
        
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                aux_losses = self._get_losses(aux_outputs, targets, indices, num_masks)
                aux_losses = {k + f"_{i}": v for k, v in aux_losses.items()}
                losses.update(aux_losses)
        
        # Apply weights to all losses
        weighted_losses = {}
        for k in losses.keys():
            if k in self.weight_dict:
                weighted_losses[k] = self.weight_dict[k] * losses[k]
        
        # Calculate component losses for tracking
        loss_ce = sum(weighted_losses[k] for k in weighted_losses if 'ce' in k)
        loss_mask = sum(weighted_losses[k] for k in weighted_losses if 'mask' in k)
        loss_dice = sum(weighted_losses[k] for k in weighted_losses if 'dice' in k)
        
        # Compute final loss
        loss = loss_ce + loss_mask + loss_dice
        
        # Save total loss for summarization
        self.buffer.append(loss.detach().item())
        
        return loss
    
    def _get_binary_mask(self, target):
        """Convert target tensor to one-hot encoding"""
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot
    
    def _get_targets(self, gt_masks):
        """
        Convert change maps to instance format expected by matcher and loss functions
        
        Args:
            gt_masks: Tensor of shape [B, H, W] containing the change maps
            
        Returns:
            List of dictionaries, each with 'labels' and 'masks' for instance segmentation
        """
        targets = []
        for mask in gt_masks:
            binary_masks = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[cls_label > 0]  # Skip background class
            if len(labels) > 0:
                binary_masks = binary_masks[labels]
            else:
                # No changes detected, create empty tensors
                binary_masks = torch.zeros((0, mask.size(0), mask.size(1)), device=mask.device)
                labels = torch.tensor([], dtype=torch.int64, device=mask.device)
            targets.append({'masks': binary_masks, 'labels': labels})
        return targets

    def _get_losses(self, outputs, targets, indices, num_masks):
        """
        Compute all the requested losses.
        """
        losses = {}
        
        # Classification loss
        loss_ce = self._get_src_classification_loss(outputs, targets, indices)
        losses["loss_ce"] = loss_ce
        
        # Mask losses
        loss_mask, loss_dice = self._get_mask_losses(outputs, targets, indices, num_masks)
        losses["loss_mask"] = loss_mask
        losses["loss_dice"] = loss_dice
        
        return losses
        
    def _get_src_classification_loss(self, outputs, targets, indices):
        """Classification loss (NLL) targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return loss_ce

    def _get_mask_losses(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss."""
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        
        # Extract target masks directly
        target_masks = []
        for t, (_, i) in zip(targets, indices):
            if "masks" in t and len(i) > 0:
                target_masks.append(t["masks"][i])
                
        # Handle case with no valid masks
        if len(target_masks) == 0:
            return torch.tensor(0.0, device=src_masks.device), torch.tensor(0.0, device=src_masks.device)
        
        # Stack masks for processing
        target_masks = torch.cat(target_masks)
        target_masks = target_masks.to(src_masks.device)
        
        # Ensure target masks have the same spatial dimensions as source masks
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=src_masks.shape[-2:],
                mode="nearest"
            ).squeeze(1)

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        
        loss_mask = sigmoid_ce_loss(src_masks, target_masks, num_masks)
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        
        return loss_mask, loss_dice

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
