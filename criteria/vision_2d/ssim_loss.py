import math
import torch
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion
from typing import Optional


def gaussian(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size (int): Size of the Gaussian window.
        sigma (float): Standard deviation of the Gaussian.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel.
    """
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ], dtype=torch.float32, device=device)
    return gauss / gauss.sum()


def create_window(window_size: int, channels: int, device: torch.device) -> torch.Tensor:
    """
    Creates a 2D Gaussian window for SSIM computation.

    Args:
        window_size (int): Size of the Gaussian window.
        channels (int): Number of channels.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: 4D Gaussian window tensor with shape (channels, 1, window_size, window_size).
    """
    _1D_window = gaussian(window_size, sigma=1.5, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channels: int,
    C1: float,
    C2: float,
) -> torch.Tensor:
    """
    Computes the SSIM index between two images.

    Args:
        img1 (torch.Tensor): First image tensor with shape (N, C, H, W).
        img2 (torch.Tensor): Second image tensor with shape (N, C, H, W).
        window (torch.Tensor): Precomputed Gaussian window.
        window_size (int): Size of the Gaussian window.
        channels (int): Number of channels.
        C1 (float): Stability constant for luminance comparison.
        C2 (float): Stability constant for contrast comparison.

    Returns:
        torch.Tensor: SSIM values with shape (N,).
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3])


class SSIMLoss(SingleTaskCriterion):
    """
    Structural Similarity Index (SSIM) Loss.
    """
    def __init__(
        self,
        window_size: Optional[int] = 11,
        channels: Optional[int] = 1,
        reduction: Optional[str] = "mean",
        device: Optional[torch.device] = torch.device('cuda'),
        C1: Optional[float] = 0.01 ** 2,
        C2: Optional[float] = 0.03 ** 2,
    ) -> None:
        """
        Args:
            window_size (Optional[int]): Size of the Gaussian window. Default is 11.
            channels (Optional[int]): Number of channels in the input images. Default is 1.
            reduction (Optional[str]): Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default is 'mean'.
            device (Optional[torch.device]): Device to place the Gaussian window on. Default is CUDA.
            C1 (Optional[float]): Stability constant for luminance comparison. Default is 0.01^2.
            C2 (Optional[float]): Stability constant for contrast comparison. Default is 0.03^2.
        """
        super().__init__()

        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number")

        self.window_size = window_size
        self.channels = channels
        self.reduction = reduction
        self.C1 = C1
        self.C2 = C2

        self.register_buffer("window", create_window(window_size, channels, device=device))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes the SSIM loss between two images.

        Args:
            img1 (torch.Tensor): First image tensor with shape (N, C, H, W).
            img2 (torch.Tensor): Second image tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: SSIM loss value.
        """
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError("Input images must have shape (N, C, H, W)")
        if img1.size(1) != self.channels or img2.size(1) != self.channels:
            raise ValueError(f"Input images must have {self.channels} channels")

        ssim_vals = compute_ssim(img1, img2, self.window, self.window_size, self.channels, self.C1, self.C2)

        if self.reduction == "mean":
            return ssim_vals.mean()
        elif self.reduction == "sum":
            return ssim_vals.sum()
        elif self.reduction == "none":
            return ssim_vals
        else:
            raise ValueError("Invalid reduction type. Choose from 'mean', 'sum', or 'none'.")
