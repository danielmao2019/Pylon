from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud


class GaussianPosNoise(BaseTransform):
    """Add Gaussian noise to point cloud positions.
    
    Args:
        std: Standard deviation of the Gaussian noise (default: 0.01)
    """

    def __init__(self, std: float = 0.01) -> None:
        super(GaussianPosNoise, self).__init__()
        assert isinstance(std, (int, float)), f"{type(std)=}"
        assert std >= 0, f"{std=}"
        self.std = std

    def _call_single(self, pc: Dict[str, Any], generator: torch.Generator) -> Dict[str, Any]:
        check_point_cloud(pc)
        
        # Validate generator device type matches point cloud device type
        assert generator.device.type == pc['pos'].device.type, (
            f"Generator device type '{generator.device.type}' must match point cloud device type '{pc['pos'].device.type}'"
        )
        
        if self.std > 0:
            noise = torch.randn(
                pc['pos'].shape, 
                device=pc['pos'].device, 
                generator=generator,
                dtype=pc['pos'].dtype
            ) * self.std
            pc['pos'] = pc['pos'] + noise
        return pc
