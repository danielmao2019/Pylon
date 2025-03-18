import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from criteria.wrappers import SingleTaskCriterion
from utils.matcher import HungarianMatcher
from criteria.vision_2d import DiceLoss


def compute_class_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute classification cost for Hungarian matching.
    For change detection, we use the negative probability of the change class.
    """
    out_prob = outputs["pred_logits"].softmax(-1)  # [num_queries, num_classes+1]
    # Use class 1 (change class) probability
    # For each query, we calculate cost for assigning it to the target
    cost_class = -out_prob[:, 1].unsqueeze(1)  # [num_queries, 1]
    return cost_class


def compute_mask_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute mask BCE cost for Hungarian matching.
    """
    pred_masks = outputs["pred_masks"]  # [num_queries, H, W]
    tgt_masks = targets["masks"]  # [num_targets, H, W] (typically just one mask for change)
    
    # Flatten spatial dimensions
    pred_masks = pred_masks.flatten(1)  # [num_queries, H*W]
    tgt_masks = tgt_masks.flatten(1)    # [num_targets, H*W]
    
    # Compute cost matrix [num_queries, num_targets]
    cost_mask = torch.zeros((pred_masks.shape[0], tgt_masks.shape[0]), device=pred_masks.device)
    
    for i in range(tgt_masks.shape[0]):
        # For each target, compute BCE with all queries
        target_i = tgt_masks[i:i+1]  # [1, H*W]
        cost_mask[:, i] = F.binary_cross_entropy_with_logits(
            pred_masks,
            target_i.expand(pred_masks.shape[0], -1),
            reduction='none'
        ).mean(dim=1)
    
    return cost_mask


def compute_dice_cost(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute dice cost for Hungarian matching.
    """
    pred_masks = outputs["pred_masks"]  # [num_queries, H, W]
    tgt_masks = targets["masks"]  # [num_targets, H, W]
    
    # Flatten spatial dimensions
    pred_masks = pred_masks.flatten(1)  # [num_queries, H*W]
    tgt_masks = tgt_masks.flatten(1)    # [num_targets, H*W]
    
    # Compute cost matrix [num_queries, num_targets]
    cost_dice = torch.zeros((pred_masks.shape[0], tgt_masks.shape[0]), device=pred_masks.device)
    
    for i in range(tgt_masks.shape[0]):
        # For each target, compute dice with all queries
        target_i = tgt_masks[i]  # [H*W]
        
        # Apply sigmoid to predicted masks
        pred_masks_prob = torch.sigmoid(pred_masks)
        
        # Compute intersection and union for dice coefficient
        numerator = 2 * (pred_masks_prob * target_i.unsqueeze(0)).sum(dim=1)
        denominator = pred_masks_prob.sum(dim=1) + target_i.sum() + 1e-7
        dice_coef = numerator / denominator
        cost_dice[:, i] = 1 - dice_coef
    
    return cost_dice


def dice_loss(pred_masks: torch.Tensor, target_masks: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the DICE loss between predicted masks and target masks.
    
    Args:
        pred_masks: Predicted mask probabilities after sigmoid, shape [N, H, W]
        target_masks: Target masks, shape [N, H, W]
        valid_mask: Optional mask of valid pixels, shape [H, W]
        
    Returns:
        Scalar DICE loss
    """
    # Apply valid mask if provided
    if valid_mask is not None:
        # Expand valid_mask to match pred_masks shape
        valid_mask_expanded = valid_mask.unsqueeze(0).expand_as(pred_masks)
        
        # Apply mask: set invalid regions to 0 in both pred and target
        pred_masks = pred_masks * valid_mask_expanded
        target_masks = target_masks * valid_mask_expanded
    
    # Flatten spatial dimensions
    pred_flat = pred_masks.flatten(1)  # [N, H*W]
    target_flat = target_masks.flatten(1)  # [N, H*W]
    
    # Compute intersection and union
    intersection = 2.0 * (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-7
    
    # Compute dice coefficient and loss
    dice_coef = intersection / denominator
    dice_loss = 1.0 - dice_coef
    
    # Average across batch
    return dice_loss.mean()


class CDMaskFormerCriterion(SingleTaskCriterion):
    """
    Criterion for CDMaskFormer that combines mask classification and mask prediction losses.
    This criterion follows the Mask2Former approach with Hungarian matching.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        ignore_value: int = 255,
        class_weight: float = 2.0,
        dice_weight: float = 5.0,
        mask_weight: float = 5.0,
        no_object_weight: float = 0.1
    ) -> None:
        """
        Initialize the CDMaskFormer criterion.
        
        Args:
            num_classes: Number of classes for semantic segmentation (default is 1 for binary change detection)
            ignore_value: Value to ignore in loss computation (typically 255 for unlabeled regions)
            class_weight: Weight for the classification loss component
            dice_weight: Weight for the dice loss component
            mask_weight: Weight for the mask/BCE loss component
            no_object_weight: Weight for the no-object class
        """
        super(CDMaskFormerCriterion, self).__init__()
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.class_weight = class_weight
        self.dice_weight = dice_weight
        self.mask_weight = mask_weight
        self.no_object_weight = no_object_weight
        
        # Initialize matcher with cost functions
        cost_functions = {
            "class": (compute_class_cost, class_weight),
            "mask": (compute_mask_cost, mask_weight),
            "dice": (compute_dice_cost, dice_weight)
        }
        self.matcher = HungarianMatcher(cost_functions=cost_functions)
        
        # Create class weights with higher weight for no-object class
        self.register_buffer(
            "empty_weight", 
            torch.ones(self.num_classes + 1)
        )
        self.empty_weight[0] = self.no_object_weight
        
    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for CDMaskFormer.
        
        Args:
            y_pred: Dictionary containing outputs from the model with keys 'pred_logits', 'pred_masks', etc.
            y_true: Dictionary containing ground truth with keys 'labels'
            
        Returns:
            Scalar loss tensor
        """
        outputs = y_pred
        targets = y_true
        
        # Check if there's anything to match
        device = outputs["pred_logits"].device
        valid_mask = targets["labels"] != self.ignore_value
        if not valid_mask.any():
            loss = torch.tensor(0.0, device=device)
            self.buffer.append(loss)
            return loss
        
        # Format targets for the matcher
        formatted_targets = []
        for b in range(targets["labels"].shape[0]):
            if valid_mask[b].any():
                # Convert to binary masks: 1 for change, 0 for no change
                bin_mask = (targets["labels"][b] > 0).float()
                # For change detection, we have a single target mask per image
                formatted_targets.append({
                    "labels": torch.tensor([1], device=device),  # Class 1 for change
                    "masks": bin_mask.unsqueeze(0)  # [1, H, W]
                })
            else:
                # Empty target for this batch element
                formatted_targets.append({
                    "labels": torch.tensor([], device=device),
                    "masks": torch.zeros((0, targets["labels"].shape[1], targets["labels"].shape[2]), device=device)
                })
        
        # Find matches between queries and targets
        indices = self.matcher(outputs, formatted_targets)
        
        # Compute classification loss
        pred_logits = outputs["pred_logits"]  # [bs, num_queries, num_classes+1]
        bs, num_queries = pred_logits.shape[:2]
        
        # Initialize target classes with -1 (no object class)
        target_classes = torch.full((bs, num_queries), -1, dtype=torch.int64, device=device)
        
        # For each batch element, set the matched queries to class 1 (change)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                # Get the target classes for the matched queries
                # In change detection, all targets are class 1 (change)
                target_classes[b, src_idx] = formatted_targets[b]["labels"][tgt_idx]
        
        # For cross-entropy, we need to map -1 to 0 (background class)
        # since cross_entropy expects targets in [0, C-1]
        target_classes_for_ce = target_classes.clone()
        target_classes_for_ce[target_classes_for_ce == -1] = 0
        
        # Compute cross-entropy loss directly
        # Flatten predictions and targets
        pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])  # [bs*num_queries, num_classes+1]
        target_classes_flat = target_classes_for_ce.view(-1)  # [bs*num_queries]
        
        loss_ce = F.cross_entropy(
            pred_logits_flat, 
            target_classes_flat,
            weight=self.empty_weight
        )
        
        # Compute mask loss and dice loss only for matched queries
        loss_mask = 0
        loss_dice = 0
        total_matched = 0
        
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if not valid_mask[b].any() or len(src_idx) == 0:
                continue
                
            # Get matched query predictions and target masks
            pred_mask = outputs["pred_masks"][b, src_idx]  # [num_matched, H, W]
            tgt_mask = formatted_targets[b]["masks"][tgt_idx]  # [num_matched, H, W]
            
            # Only compute loss for valid pixels
            valid_b = valid_mask[b]
            if valid_b.any():
                # Apply sigmoid for predicted masks
                pred_mask_prob = torch.sigmoid(pred_mask)
                
                # Prepare binary target mask
                binary_target = tgt_mask.bool().float()  # [num_matched, H, W]
                
                # Use BCE for mask loss - only on valid pixels
                # First, mask out invalid pixels
                valid_mask_expanded = valid_b.unsqueeze(0).expand_as(binary_target)
                loss_mask += F.binary_cross_entropy_with_logits(
                    pred_mask[valid_mask_expanded],
                    binary_target[valid_mask_expanded]
                )
                
                # Use our custom dice_loss function
                loss_dice += dice_loss(
                    pred_mask_prob,  # Already sigmoidized
                    binary_target,   # Binary targets
                    valid_b          # Valid pixel mask
                )
                
                total_matched += 1
        
        # Normalize the losses
        if total_matched > 0:
            loss_mask /= total_matched
            loss_dice /= total_matched
        
        # Compute total loss
        loss = self.class_weight * loss_ce + self.mask_weight * loss_mask + self.dice_weight * loss_dice
        
        # Store in buffer and return
        self.buffer.append(loss)
        return loss
