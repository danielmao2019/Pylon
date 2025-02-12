import math
import torch
import torch.nn.functional as F
from criteria.wrappers import SingleTaskCriterion


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Generates a 1D Gaussian kernel.
    
    Args:
        window_size (int): The size of the window.
        sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel.
    """
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size: int, channels: int) -> torch.Tensor:
    """
    Creates a 2D Gaussian window for SSIM computation.
    
    Args:
        window_size (int): Size of the Gaussian window.
        channels (int): Number of input channels.
    
    Returns:
        torch.Tensor: A 4D tensor representing the Gaussian window for convolution.
    """
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channels, 1, window_size, window_size).contiguous()

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channels: int, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (torch.Tensor): First input image of shape (N, C, H, W).
        img2 (torch.Tensor): Second input image of shape (N, C, H, W).
        window (torch.Tensor): Precomputed Gaussian window.
        window_size (int): Size of the Gaussian window.
        channels (int): Number of channels in the images.
        size_average (bool): Whether to average the SSIM map.
    
    Returns:
        torch.Tensor: SSIM score.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))

class SSIMLoss(SingleTaskCriterion):
    """
    SSIM-based loss function for image similarity.
    """
    def __init__(self, window_size: int = 11, channels: int = 1, size_average: bool = True):
        """
        Initializes SSIMLoss with a precomputed Gaussian window.
        
        Args:
            window_size (int): Size of the Gaussian window.
            channels (int): Number of channels in input images.
            size_average (bool): Whether to average the SSIM map.
        """
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.size_average = size_average
        self.window = create_window(window_size, channels)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Computes the SSIM loss between two images.
        
        Args:
            img1 (torch.Tensor): First input image of shape (N, C, H, W).
            img2 (torch.Tensor): Second input image of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: SSIM loss value.
        """
        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError("Input images must have shape (N, C, H, W)")
        if img1.size(1) != self.channels or img2.size(1) != self.channels:
            raise ValueError(f"Input images must have {self.channels} channels")
        
        # Move window to the same device and type as img1
        window = self.window.to(img1.device).type_as(img1)
        
        return compute_ssim(img1, img2, window, self.window_size, self.channels, self.size_average)
