from typing import Optional
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class SSIMLoss(DenseClassificationCriterion):
    """
    Structural Similarity Index (SSIM) Loss for semantic segmentation.

    SSIM measures the similarity between two images based on luminance, contrast, and structure.
    This implementation adapts SSIM for segmentation by applying it to one-hot encoded predictions
    and ground truth.

    This implementation supports:
    - Window size customization
    - Class weights to handle class imbalance
    - Ignore value to exclude specific pixel values from loss computation

    Attributes:
        ignore_value (int): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        class_weights (Optional[torch.Tensor]): Optional weights for each class
        window_size (int): Size of the Gaussian window
    """

    def __init__(
        self,
        ignore_value: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
        window_size: int = 11,
        **kwargs,
    ) -> None:
        """
        Initialize the criterion.

        Args:
            ignore_value: Value to ignore in loss computation (usually background/unlabeled pixels).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
            window_size: Size of the Gaussian window for SSIM computation.
        """
        super(SSIMLoss, self).__init__(
            ignore_value=ignore_value,
            reduction=reduction,
            class_weights=class_weights,
            **kwargs,
        )

        # Create Gaussian window
        window = self._create_window(window_size)
        self.register_buffer('window', window)

    def _create_window(self, window_size: int) -> torch.Tensor:
        """
        Create a 2D Gaussian window.

        Args:
            window_size: Size of the window (must be odd)

        Returns:
            2D Gaussian window tensor of shape (1, 1, window_size, window_size)
        """
        assert window_size % 2 == 1, f"Window size must be odd, got {window_size}"

        # Create 1D Gaussian window
        sigma = 1.5
        gauss = torch.exp(
            -torch.pow(torch.linspace(-(window_size//2), window_size//2, window_size), 2.0) / (2.0 * sigma * sigma)
        )
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian window
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)  # (window_size, window_size)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

        return window

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to SSIM loss.

        Args:
            y_pred: Predicted logits tensor of shape (N, C, H, W)
            y_true: Ground truth labels tensor of shape (N, H, W)

        Raises:
            ValueError: If validation fails
        """
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid pixels (not equal to ignore_value).

        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)

        Returns:
            Boolean tensor of shape (N, H, W), True for valid pixels

        Raises:
            ValueError: If all pixels in target are ignored
        """
        valid_mask = (y_true != self.ignore_value)
        if valid_mask.sum() == 0:
            raise ValueError("All pixels in target are ignored. Cannot compute loss.")
        return valid_mask  # (N, H, W)

    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W) probabilities
        y_true: torch.Tensor,  # (N, C, H, W) one-hot encoded
        valid_mask: torch.Tensor,  # (N, 1, H, W) unsqueezed valid mask
    ) -> torch.Tensor:  # (N, C)
        """
        Compute SSIM loss for each class and sample in the batch.

        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels

        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        # Apply valid mask by broadcasting to all channels
        y_pred = y_pred * valid_mask
        y_true = y_true * valid_mask

        # Compute means
        mu_pred = F.conv2d(y_pred, self.window.expand(y_pred.size(1), -1, -1, -1),
                          padding=self.window.size(-1)//2, groups=y_pred.size(1))  # (N, C, H, W)
        mu_true = F.conv2d(y_true, self.window.expand(y_true.size(1), -1, -1, -1),
                          padding=self.window.size(-1)//2, groups=y_true.size(1))  # (N, C, H, W)

        # Compute variances and covariance
        mu_pred_sq = mu_pred.pow(2)
        mu_true_sq = mu_true.pow(2)
        mu_pred_true = mu_pred * mu_true

        sigma_pred_sq = F.conv2d(y_pred * y_pred, self.window.expand(y_pred.size(1), -1, -1, -1),
                                padding=self.window.size(-1)//2, groups=y_pred.size(1)) - mu_pred_sq
        sigma_true_sq = F.conv2d(y_true * y_true, self.window.expand(y_true.size(1), -1, -1, -1),
                                padding=self.window.size(-1)//2, groups=y_true.size(1)) - mu_true_sq
        sigma_pred_true = F.conv2d(y_pred * y_true, self.window.expand(y_pred.size(1), -1, -1, -1),
                                  padding=self.window.size(-1)//2, groups=y_pred.size(1)) - mu_pred_true

        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute SSIM
        numerator = (2 * mu_pred_true + C1) * (2 * sigma_pred_true + C2)
        denominator = (mu_pred_sq + mu_true_sq + C1) * (sigma_pred_sq + sigma_true_sq + C2)
        ssim_map = numerator / denominator  # (N, C, H, W)

        # Average over spatial dimensions for valid pixels only
        valid_pixels = valid_mask.sum(dim=(2, 3))  # (N, 1)
        ssim_per_class = (ssim_map * valid_mask).sum(dim=(2, 3)) / (valid_pixels + 1e-8)  # (N, C)

        # Return SSIM loss
        return 1 - ssim_per_class
