from typing import Tuple, Optional
import torch
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_point_cloud_segmentation


class PointCloudSegmentationCriterion(SingleTaskCriterion):
    """
    Criterion for 3D point cloud segmentation tasks.
    
    This criterion computes the cross-entropy loss between predicted class logits
    and ground truth labels for each point in the point cloud.
    
    Attributes:
        criterion: The underlying PyTorch loss function (CrossEntropyLoss).
        class_weights: Optional tensor of weights for each class (registered as buffer).
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in the loss computation (usually for background/unlabeled points).
            class_weights: Optional weights for each class to address class imbalance.
        """
        super(PointCloudSegmentationCriterion, self).__init__()
        if ignore_index is None:
            ignore_index = -100  # PyTorch's default ignore index
        
        # Register class weights as a buffer if provided
        self.register_buffer('class_weights', None)
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            self.register_buffer(
                'class_weights',
                torch.tensor(class_weights, dtype=torch.float32)
            )
        
        # Create criterion
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=self.class_weights, reduction='mean',
        )
        
        # Verify weights were set correctly
        if self.class_weights is not None:
            assert torch.allclose(self.criterion.weight, self.class_weights), \
                f"Weights not set correctly: {self.criterion.weight} != {self.class_weights}"

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss for point cloud segmentation.
        
        Args:
            y_pred (torch.Tensor): A float32 tensor of shape [N, C] for predicted logits,
                                   where N is the total number of points (possibly from multiple samples)
                                   and C is the number of classes.
            y_true (torch.Tensor): An int64 tensor of shape [N] for ground-truth labels.

        Returns:
            loss (torch.Tensor): A float32 scalar tensor for loss value.
        """
        # Input checks
        check_point_cloud_segmentation(y_pred=y_pred, y_true=y_true)
        
        # Compute loss
        return self.criterion(input=y_pred, target=y_true)
