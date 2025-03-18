import torch
import torch.nn.functional as F
from typing import Dict, List, Union
from criteria.wrappers import SingleTaskCriterion
from utils.matcher import HungarianMatcher
from models.change_detection.cdmaskformer.utils.nested_tensor import nested_tensor_from_tensor_list


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


class CDMaskFormerCriterion(SingleTaskCriterion):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and outputs
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, class_weight=2, mask_weight=5, dice_weight=5, 
                 no_object_weight=0.1, dec_layers=10):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.no_object_weight = no_object_weight
        self.dec_layers = dec_layers
        
        # Setup cost functions for matcher
        cost_class = lambda outputs, targets: -outputs["pred_logits"].softmax(-1)[:, targets["labels"]]
        
        cost_functions = {
            "cost_class": (cost_class, class_weight),
            "cost_mask": (lambda o, t: batch_sigmoid_focal_loss(o["pred_masks"].flatten(1), t["masks"].flatten(1)), mask_weight),
            "cost_dice": (lambda o, t: batch_dice_loss(o["pred_masks"].flatten(1), t["masks"].flatten(1)), dice_weight)
        }
        
        self.matcher = HungarianMatcher(cost_functions=cost_functions)
        
        # Create weight dict for losses
        self.weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
        self.weight_dict.update(aux_weight_dict)
        
        # Create class weights with higher weight for no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = no_object_weight
        self.register_buffer('empty_weight', empty_weight)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        """
        Perform forward pass through the criterion.
        
        Args:
            y_pred: Dict with model predictions
            y_true: Ground truth targets, can be a Dict with 'change_map' or a List of Dicts with instance targets
            
        Returns:
            Total loss (scalar tensor)
        """
        # Handle input formats - convert dict input to list of dicts if needed
        if isinstance(y_true, dict):
            # Extract the change map
            if 'change_map' in y_true:
                change_map = y_true['change_map']
                
                # Convert to list of dictionaries with 'labels' and 'masks'
                formatted_targets = []
                batch_size = change_map.shape[0]
                
                for b in range(batch_size):
                    batch_mask = change_map[b]
                    
                    # Convert semantic segmentation to instance segmentation
                    if torch.any(batch_mask == 1):
                        # Create a binary mask for change
                        binary_mask = (batch_mask == 1).float().unsqueeze(0)
                        
                        # Create labels tensor (1 for change class)
                        labels = torch.tensor([1], dtype=torch.int64, device=batch_mask.device)
                        
                        formatted_targets.append({
                            'labels': labels,
                            'masks': binary_mask
                        })
                    else:
                        # No change regions
                        formatted_targets.append({
                            'labels': torch.tensor([], dtype=torch.int64, device=batch_mask.device),
                            'masks': torch.zeros((0, batch_mask.shape[0], batch_mask.shape[1]), 
                                               dtype=torch.float, device=batch_mask.device)
                        })
                
                # Update y_true to use the formatted targets
                y_true = formatted_targets
            else:
                # Handle other dictionary formats if needed
                batch_size = y_pred['pred_logits'].shape[0]
                device = y_pred['pred_logits'].device
                y_true = [{'labels': torch.tensor([], dtype=torch.int64, device=device), 
                           'masks': torch.zeros((0, 1, 1), dtype=torch.float, device=device)} 
                         for _ in range(batch_size)]
        
        outputs = y_pred.copy()
        
        # Upsample masks for processing
        outputs["pred_masks"] = F.interpolate(
            outputs["pred_masks"],
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=False,
        )
        
        # Handle auxiliary outputs the same way
        if "aux_outputs" in outputs:
            for aux_outputs in outputs["aux_outputs"]:
                aux_outputs["pred_masks"] = F.interpolate(
                    aux_outputs["pred_masks"],
                    scale_factor=(4, 4),
                    mode="bilinear",
                    align_corners=False,
                )
        
        # Compute the losses
        losses = {}
        
        # Get the indices for matching 
        indices = self.matcher(outputs, y_true)
        
        # Calculate number of masks for normalization
        num_masks = sum(len(t["labels"]) for t in y_true)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs["pred_logits"].device)
        num_masks = max(num_masks.item(), 1)  # Avoid division by zero
        
        # Main loss computation
        losses.update(self._get_losses(outputs, y_true, indices, num_masks))
        
        # Auxiliary loss computation
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, y_true)
                aux_losses = self._get_losses(aux_outputs, y_true, aux_indices, num_masks)
                losses.update({f"{k}_{i}": v for k, v in aux_losses.items()})
                
        # Compute weighted sum of all losses
        weighted_losses = {k: self.weight_dict[k] * losses[k] for k in losses.keys() if k in self.weight_dict}
        total_loss = sum(weighted_losses.values())
        
        # Store for tracking
        self.buffer.append(total_loss.detach().cpu())
        
        return total_loss
    
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
        
        masks = [t["masks"] for t in targets]
        
        # Use nested tensor from tensor list to handle different-sized targets
        target_masks = nested_tensor_from_tensor_list(masks)
        if isinstance(target_masks, tuple):
            target_masks = target_masks[0]
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        
        loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_masks)
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        
        return loss_mask, loss_dice

    def _get_src_permutation_idx(self, indices):
        """
        Given a list of tuples (src_idx, tgt_idx) for each batch element, 
        returns the permutation indices for the source.
        """
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        Given a list of tuples (src_idx, tgt_idx) for each batch element, 
        returns the permutation indices for the target.
        """
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
