from typing import Optional, Union, Sequence
import torch
import torchvision.transforms as T
from data.transforms.base_transform import BaseTransform


class ColorJitter(BaseTransform):
    """Deterministic ColorJitter that accepts seed parameter.
    
    This wrapper around torchvision.transforms.ColorJitter ensures deterministic behavior
    by properly handling the seed parameter and controlling PyTorch's global random state.
    """
    
    def __init__(
        self,
        brightness: Union[float, Sequence[float]] = 0,
        contrast: Union[float, Sequence[float]] = 0,
        saturation: Union[float, Sequence[float]] = 0,
        hue: Union[float, Sequence[float]] = 0
    ) -> None:
        """Initialize ColorJitter transform.
        
        Args:
            brightness: How much to jitter brightness. brightness_factor is chosen
                uniformly from [max(0, 1 - brightness), 1 + brightness] or the given
                [min, max]. Should be non negative numbers.
            contrast: How much to jitter contrast. contrast_factor is chosen
                uniformly from [max(0, 1 - contrast), 1 + contrast] or the given
                [min, max]. Should be non negative numbers.
            saturation: How much to jitter saturation. saturation_factor is chosen
                uniformly from [max(0, 1 - saturation), 1 + saturation] or the given
                [min, max]. Should be non negative numbers.
            hue: How much to jitter hue. hue_factor is chosen uniformly from
                [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or
                -0.5 <= min <= max <= 0.5.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
        # Create the underlying torchvision ColorJitter
        self._transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def _call_single(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply ColorJitter with deterministic seeding.
        
        Args:
            image: Input tensor image
            generator: Optional generator for deterministic behavior
            
        Returns:
            Color jittered image tensor
        """
        if generator is not None:
            # Save current global random state
            current_state = torch.get_rng_state()
            
            try:
                # Use the generator's initial seed for consistent behavior
                # This ensures the same seed always produces the same result
                seed = generator.initial_seed()
                torch.manual_seed(seed)
                
                # Apply ColorJitter with controlled random state
                result = self._transform(image)
            finally:
                # Always restore original global state
                torch.set_rng_state(current_state)
            
            return result
        else:
            # No generator provided, use default behavior
            return self._transform(image)
