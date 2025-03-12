from typing import Tuple, Optional
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
        class_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in loss computation. If None, child classes should
                         provide a default value appropriate for their task.
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
        """
        super(DenseClassificationCriterion, self).__init__(ignore_index=ignore_index)
        
        # Register class weights as a buffer if provided
        self.register_buffer('class_weights', None)
        if class_weights is not None:
            self._set_class_weights(class_weights)
            
    def _set_class_weights(self, class_weights: Tuple[float, ...]) -> None:
        """
        Set and validate class weights.
        
        Args:
            class_weights: Tuple of weights for each class
            
        Raises:
            ValueError: If weights are invalid
        """
        if not isinstance(class_weights, tuple):
            raise ValueError(f"class_weights must be a tuple, got {type(class_weights)}")
            
        if not all(isinstance(w, float) for w in class_weights):
            raise ValueError("All class weights must be floats")
            
        if not all(w >= 0 for w in class_weights):
            raise ValueError("All class weights must be non-negative")
            
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize to sum to 1
        self.register_buffer('class_weights', weights_tensor)
        
    def _to_probabilities(self, logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Convert logits to probabilities using softmax.
        
        Args:
            logits: Raw model outputs (pre-softmax)
            dim: Dimension along which to apply softmax (usually the class dimension)
            
        Returns:
            Tensor of probabilities
        """
        return torch.nn.functional.softmax(logits, dim=dim)
        
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
        
    def _compute_class_weights(
        self,
        y_true: torch.Tensor,
        num_classes: int,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class weights based on the frequency of each class in the ground truth.
        
        Args:
            y_true: Ground truth labels
            num_classes: Number of classes
            valid_mask: Boolean mask of valid pixels
            
        Returns:
            Tensor of class weights
        """
        with torch.no_grad():
            # Count instances of each class in valid pixels
            class_counts = torch.bincount(
                y_true[valid_mask].view(-1),
                minlength=num_classes
            ).float()
            
            # Compute inverse frequency weights
            total_pixels = valid_mask.sum()
            weights = (total_pixels - class_counts) / total_pixels
            
            # Handle edge cases
            weights = torch.clamp(weights, min=0.0)  # Ensure non-negative
            weights[class_counts == 0] = 0  # Zero weight for missing classes
            
            # Normalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
                
            return weights
