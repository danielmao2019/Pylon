from typing import Optional
import math
import torch
import torch.nn.functional as F
from criteria.wrappers.dense_prediction_criterion import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size (int): Size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel.
    """
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ], dtype=torch.float32)
    return gauss / gauss.sum()


def create_window(window_size: int, num_classes: int) -> torch.Tensor:
    """
    Creates a 2D Gaussian window for SSIM computation.

    Args:
        window_size (int): Size of the Gaussian window.
        num_classes (int): Number of classes for semantic segmentation.

    Returns:
        torch.Tensor: 4D Gaussian window tensor with shape (num_classes, 1, window_size, window_size).
    """
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(num_classes, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    num_classes: int,
    C1: float,
    C2: float,
) -> torch.Tensor:
    """
    Computes the SSIM loss between two images.

    Args:
        img1 (torch.Tensor): First image tensor with shape (N, C, H, W).
        img2 (torch.Tensor): Second image tensor with shape (N, C, H, W).
        window (torch.Tensor): Precomputed Gaussian window.
        window_size (int): Size of the Gaussian window.
        num_classes (int): Number of classes for semantic segmentation.
        C1 (float): Stability constant for luminance comparison.
        C2 (float): Stability constant for contrast comparison.

    Returns:
        torch.Tensor: SSIM loss values with shape (N, C).
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=num_classes)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=num_classes)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=num_classes) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=num_classes) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=num_classes) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Convert similarity to loss and average over spatial dimensions
    return 1 - ssim_map.mean(dim=[2, 3])  # (N, C)


class SSIMLoss(DenseClassificationCriterion):
    """
    Structural Similarity Index (SSIM) Loss for semantic segmentation.
    
    This criterion computes the SSIM loss between predicted class probabilities
    and one-hot encoded ground truth labels for each class.
    
    Attributes:
        ignore_index: Index to ignore in the loss computation.
        num_classes: Number of classes for semantic segmentation.
        window_size: Size of the Gaussian window for SSIM computation.
        C1: Stability constant for luminance comparison.
        C2: Stability constant for contrast comparison.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
        reduction: str = 'mean',
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            num_classes: Number of classes for semantic segmentation.
            ignore_index: Index to ignore in the loss computation. Defaults to 255.
            window_size: Size of the Gaussian window. Must be odd. Defaults to 11.
            C1: Stability constant for luminance comparison. Defaults to 0.01^2.
            C2: Stability constant for contrast comparison. Defaults to 0.03^2.
            reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
        """
        super(SSIMLoss, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
        )

        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")

        self.num_classes = num_classes
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        # Create and register window
        window = create_window(window_size, num_classes=num_classes)
        self.register_buffer('window', window)

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to semantic segmentation.
        
        Args:
            y_pred: Predicted logits tensor of shape (N, C, H, W)
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        assert y_pred.size(1) == self.window.size(0), (
            f"Number of classes in prediction ({y_pred.size(1)}) must match "
            f"number of classes in window ({self.window.size(0)})"
        )

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid pixels (not equal to ignore_index).
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, 1, H, W), True for valid pixels
        """
        valid_mask = (y_true != self.ignore_index)
        if valid_mask.sum() == 0:
            raise ValueError("All pixels in target are ignored. Cannot compute loss.")
        return valid_mask.unsqueeze(1)  # (N, 1, H, W)

    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W) probabilities
        y_true: torch.Tensor,  # (N, C, H, W) one-hot encoded
        valid_mask: torch.Tensor,  # (N, 1, H, W)
    ) -> torch.Tensor:  # (N, C)
        """
        Compute SSIM loss for each class and sample.
        
        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        # Compute SSIM loss per class
        return compute_ssim(
            y_pred * valid_mask, y_true * valid_mask,
            self.window,
            self.window_size,
            self.num_classes,
            self.C1, self.C2
        )  # (N, C)
