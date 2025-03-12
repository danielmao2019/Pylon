from typing import Optional
import math
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
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
    Criterion for computing SSIM loss.
    
    This criterion computes the Structural Similarity Index (SSIM) loss between
    predicted class probabilities and ground truth labels for each pixel in the image.
    
    The SSIM loss is defined as 1 - SSIM where SSIM measures the structural similarity
    between two images based on luminance, contrast, and structure.
    
    Attributes:
        ignore_index: Index to ignore in loss computation (usually background/unlabeled pixels).
        class_weights: Optional weights for each class (registered as buffer).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        window_size: Size of the Gaussian window for SSIM computation.
        window: Gaussian window tensor (registered as buffer).
    """

    def __init__(
        self,
        ignore_index: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
        window_size: int = 11,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in loss computation (usually background/unlabeled pixels).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
            window_size: Size of the Gaussian window for SSIM computation.
        """
        super(SSIMLoss, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
            class_weights=class_weights
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
        Get mask for valid pixels (not equal to ignore_index).
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid pixels
            
        Raises:
            ValueError: If all pixels in target are ignored
        """
        valid_mask = (y_true != self.ignore_index)
        if valid_mask.sum() == 0:
            raise ValueError("All pixels in target are ignored. Cannot compute loss.")
        return valid_mask  # (N, H, W)

    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W) probabilities
        y_true: torch.Tensor,  # (N, C, H, W) one-hot encoded
        valid_mask: torch.Tensor,  # (N, 1, H, W)
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
        valid_mask = valid_mask.expand(-1, y_pred.size(1), -1, -1)  # (N, C, H, W)
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
        valid_pixels = valid_mask.sum(dim=(2, 3))  # (N, C)
        ssim_per_class = (ssim_map * valid_mask).sum(dim=(2, 3)) / (valid_pixels + 1e-8)  # (N, C)
        
        # Return SSIM loss
        return 1 - ssim_per_class
