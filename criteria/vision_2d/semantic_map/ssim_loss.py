from typing import Optional
import math
import torch
import torch.nn.functional as F
from criteria.vision_2d import SemanticMapBaseCriterion
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

    # Convert similarity to loss
    return 1 - ssim_map.mean(dim=[2, 3])


class SSIMLoss(SemanticMapBaseCriterion):
    """
    Structural Similarity Index (SSIM) Loss.
    """
    def __init__(
        self,
        num_classes: int,
        window_size: Optional[int] = 11,
        C1: Optional[float] = 0.01 ** 2,
        C2: Optional[float] = 0.03 ** 2,
        **kwargs,
    ) -> None:
        """
        Args:
            num_classes (int): Number of classes for semantic segmentation.
            window_size (Optional[int]): Size of the Gaussian window. Default is 11.
            C1 (Optional[float]): Stability constant for luminance comparison. Default is 0.01^2.
            C2 (Optional[float]): Stability constant for contrast comparison. Default is 0.03^2.
        """
        super(SSIMLoss, self).__init__(**kwargs)

        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")

        self.num_classes = num_classes
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        # Create and register window
        window = create_window(window_size, num_classes=num_classes)
        self.register_buffer('window', window)

    def _compute_semantic_map_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the SSIM loss between two images.

        Args:
            y_pred (torch.Tensor): First image tensor with shape (N, C, H, W).
            y_true (torch.Tensor): Second image tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: SSIM loss value.
        """
        # Assert number of classes matches the window
        assert y_pred.size(1) == self.window.size(0), (
            f"Number of classes in prediction ({y_pred.size(1)}) must match "
            f"number of classes in window ({self.window.size(0)})"
        )

        return compute_ssim(y_pred, y_true, self.window, self.window_size, y_pred.size(1), self.C1, self.C2)
