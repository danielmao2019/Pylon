from typing import Tuple, Optional
from abc import abstractmethod
import torch
from criteria.wrappers.dense_prediction_criterion import DensePredictionCriterion


class DenseClassificationCriterion(DensePredictionCriterion):
    """
    Base class for dense classification tasks that assign a class label to each pixel.
    
    This includes tasks like semantic segmentation, where each pixel is assigned a class label.
    The class handles common functionality like:
    - Class weights for handling class imbalance
    - Conversion between logits and probabilities
    - One-hot encoding of ground truth labels
    
    Attributes:
        ignore_index (int): Index to ignore in loss computation (e.g., for unlabeled pixels)
        class_weights (torch.Tensor): Optional weights for each class (registered as buffer)
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        num_classes: Optional[int] = None,
        class_weights: Optional[Tuple[float, ...]] = None,
        reduction: str = 'mean',
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in loss computation. If None, child classes should
                         provide a default value appropriate for their task.
            num_classes: Number of classes. Required if class_weights is None.
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        """
        super(DenseClassificationCriterion, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction
        )
        
        self._init_class_weights(class_weights, num_classes)
            
    def _init_class_weights(self, class_weights: Optional[Tuple[float, ...]], num_classes: Optional[int]) -> None:
        """
        Initialize class weights.
        
        Args:
            class_weights: Optional tuple of weights for each class. If None, uniform weights will be used.
            num_classes: Number of classes. Required if class_weights is None.
            
        Raises:
            ValueError: If weights are invalid or num_classes is missing when needed
        """
        if class_weights is None:
            if num_classes is None:
                raise ValueError("num_classes must be provided when class_weights is None")
            # Set uniform weights
            weights_tensor = torch.ones(num_classes, dtype=torch.float32) / num_classes
            self.register_buffer('class_weights', weights_tensor)
        else:
            if not isinstance(class_weights, tuple):
                raise ValueError(f"class_weights must be a tuple, got {type(class_weights)}")
                
            if not all(isinstance(w, float) for w in class_weights):
                raise ValueError("All class weights must be floats")
                
            if not all(w >= 0 for w in class_weights):
                raise ValueError("All class weights must be non-negative")
                
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize to sum to 1
            self.register_buffer('class_weights', weights_tensor)

    def _to_one_hot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Convert integer labels to one-hot encoding.
        
        Args:
            labels: Integer tensor of class labels
            num_classes: Number of classes
            
        Returns:
            One-hot encoded tensor
        """
        device = labels.device
        shape = labels.shape
        
        # Create one-hot encoding
        one_hot = torch.zeros(
            (shape[0], num_classes, *shape[1:]),
            dtype=torch.float32,
            device=device
        )
        
        # Handle ignore_index to avoid warnings
        valid_labels = labels != self.ignore_index
        one_hot.scatter_(1, labels[valid_labels].unsqueeze(1), 1)
        
        return one_hot
    def _prepare_inputs(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs by converting predictions to probabilities and targets to one-hot.
        
        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Ground truth tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Tuple of:
            - Predicted probabilities tensor of shape (N, C, H, W)
            - One-hot encoded ground truth tensor of shape (N, C, H, W)
            - Expanded valid mask of shape (N, 1, H, W)
        """
        # Check input shapes
        N, C, H, W = y_pred.shape
        assert y_true.shape == (N, H, W), \
            f"Ground truth shape mismatch: expected ({N}, {H}, {W}), got {y_true.shape}"
        assert valid_mask.shape == (N, H, W), \
            f"Valid mask shape mismatch: expected ({N}, {H}, {W}), got {valid_mask.shape}"

        # Convert logits to probabilities
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)  # (N, C, H, W)

        # Convert labels to one-hot
        y_true = self._to_one_hot(y_true, C)  # (N, C, H, W)
        assert y_true.shape == (N, C, H, W), \
            f"One-hot ground truth shape mismatch: expected ({N}, {C}, {H}, {W}), got {y_true.shape}"

        # Expand valid mask
        valid_mask = valid_mask.unsqueeze(1)  # (N, 1, H, W)
        assert valid_mask.shape == (N, 1, H, W), \
            f"Expanded valid mask shape mismatch: expected ({N}, 1, {H}, {W}), got {valid_mask.shape}"

        return y_pred, y_true, valid_mask

    @abstractmethod
    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for each class and sample in the batch.
        
        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Ground truth tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        raise NotImplementedError

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for each sample in the batch, applying class weights if provided.
        
        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Ground truth tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Convert inputs to appropriate format
        y_pred, y_true, valid_mask = self._prepare_inputs(y_pred, y_true, valid_mask)

        # Get per-class losses from subclass implementation
        per_class_loss = self._compute_per_class_loss(y_pred, y_true, valid_mask)  # (N, C)
        
        # Apply class weights
        per_class_loss = per_class_loss * self.class_weights
        return per_class_loss.sum(dim=1)  # (N,)
