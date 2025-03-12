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
        ignore_value: Value to ignore in loss computation (usually background/unlabeled points).
        class_weights: Optional tensor of weights for each class (registered as buffer).
        criterion: The underlying PyTorch loss function (CrossEntropyLoss).
    """

    def __init__(
        self,
        ignore_value: Optional[int] = None,
        class_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        """
        Initialize the criterion.

        Args:
            ignore_value: Value to ignore in the loss computation (usually background/unlabeled points).
                         Defaults to -100 (PyTorch's default ignore index).
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
        """
        super(PointCloudSegmentationCriterion, self).__init__()

        # Set default ignore_value
        if ignore_value is None:
            ignore_value = -100  # PyTorch's default ignore index
        self.ignore_value = ignore_value

        # Register class weights as a buffer if provided
        self.register_buffer('class_weights', None)
        if class_weights is not None:
            assert type(class_weights) == tuple, f"{type(class_weights)=}"
            assert all([type(elem) == float for elem in class_weights])
            assert all([w >= 0 for w in class_weights]), "Class weights must be non-negative"
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize to sum to 1
            self.register_buffer('class_weights', weights_tensor)

        # Create criterion
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_value, weight=self.class_weights, reduction='mean',
        )

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
