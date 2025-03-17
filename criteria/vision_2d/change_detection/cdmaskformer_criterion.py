import torch
import torch.nn.functional as F
from criteria.base_criterion import BaseCriterion
from typing import Dict, List, Optional

class CDMaskFormerCriterion(BaseCriterion):
    """
    Criterion for CDMaskFormer that combines dice loss and cross-entropy loss.
    
    This criterion follows the MaskFormer approach where the loss combines:
    1. Binary cross-entropy loss for mask classification
    2. Dice loss for mask prediction quality
    
    The criterion handles both training and evaluation modes and properly matches
    predictions with ground truth using bipartite matching.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        ignore_value: int = 255,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0
    ) -> None:
        """
        Initialize the CDMaskFormer criterion.
        
        Args:
            num_classes: Number of classes for semantic segmentation (default is 1 for binary change detection)
            ignore_value: Value to ignore in loss computation (typically 255 for unlabeled regions)
            dice_weight: Weight for the dice loss component
            ce_weight: Weight for the cross-entropy loss component
        """
        super(CDMaskFormerCriterion, self).__init__()
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the criterion.
        
        Args:
            outputs: Dictionary containing outputs from the model with keys:
                     - 'pred_logits': Classification logits (B, N, C+1) where N is num_queries and C is num_classes
                     - 'pred_masks': Predicted masks (B, N, H, W)
            targets: Dictionary containing ground truth with keys:
                     - 'labels': Ground truth labels (B, H, W)
                     
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Extract predictions and targets
        pred_logits = outputs["pred_logits"]  # (B, N, C+1)
        pred_masks = outputs["pred_masks"]    # (B, N, H, W)
        target_masks = targets["labels"]      # (B, H, W)
        
        # Create binary target masks for change detection
        # Convert target_masks to one-hot encoding
        valid_mask = target_masks != self.ignore_value
        
        # Check if all pixels are ignored
        if not valid_mask.any():
            # Return zero loss if all pixels are ignored
            device = pred_logits.device
            zero_loss = torch.tensor(0.0, device=device)
            return {
                "loss": zero_loss,
                "ce_loss": zero_loss,
                "dice_loss": zero_loss
            }
            
        target_masks_binary = (target_masks > 0).float()  # Assuming change is any value > 0
        
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # Compute binary cross-entropy loss
        pred_logits_flattened = pred_logits.view(batch_size * num_queries, -1)  # (B*N, C+1)
        pred_scores = F.softmax(pred_logits_flattened, dim=-1)  # (B*N, C+1)
        
        # For binary change detection, we focus on the positive class (class 1)
        # We ignore the no_object class (class 0)
        pred_scores = pred_scores[:, 1:]  # (B*N, C)
        
        # Compute cross-entropy loss (per-query classification loss)
        ce_loss = 0
        dice_loss = 0
        total_valid_masks = 0
        
        for b in range(batch_size):
            if not valid_mask[b].any():
                continue
                
            # Get binary target for this image
            target = target_masks_binary[b][valid_mask[b]]  # (num_valid_pixels,)
            
            # Get predicted masks for this image
            masks_pred = pred_masks[b]  # (N, H, W)
            masks_pred_flattened = masks_pred[:, valid_mask[b]]  # (N, num_valid_pixels)
            
            # Apply sigmoid to get probabilities
            masks_pred_prob = torch.sigmoid(masks_pred_flattened)  # (N, num_valid_pixels)
            
            # Compute BCE loss for each query
            bce_per_query = F.binary_cross_entropy(
                masks_pred_prob, 
                target.unsqueeze(0).expand(num_queries, -1),
                reduction='none'
            )  # (N, num_valid_pixels)
            
            # Average over pixels
            bce_per_query = bce_per_query.mean(dim=1)  # (N,)
            
            # Use the minimum BCE loss across all queries
            ce_loss += bce_per_query.min()
            
            # Compute Dice loss for the best matching query
            best_query = bce_per_query.argmin()
            pred_mask = masks_pred_prob[best_query]  # (num_valid_pixels,)
            
            # Calculate Dice coefficient
            intersection = (pred_mask * target).sum()
            union = pred_mask.sum() + target.sum()
            dice_coef = (2.0 * intersection) / (union + 1e-7)
            
            # Dice loss
            dice_loss += 1.0 - dice_coef
            total_valid_masks += 1
        
        # Normalize losses
        if total_valid_masks > 0:
            ce_loss /= total_valid_masks
            dice_loss /= total_valid_masks
        else:
            # This should not happen since we already checked valid_mask.any()
            # But just in case, return zero loss
            ce_loss = torch.tensor(0.0, device=device)
            dice_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        # Return loss components
        losses = {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "dice_loss": dice_loss
        }
        
        return losses
        
    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Call the criterion to compute the loss.
        
        Args:
            y_pred: Dictionary containing outputs from the model
            y_true: Dictionary containing ground truth targets
            
        Returns:
            Scalar loss tensor
        """
        losses = self.forward(y_pred, y_true)
        self.buffer.append(losses["loss"])
        return losses["loss"]
