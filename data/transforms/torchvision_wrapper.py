from typing import Optional, Any, Type
import torch
import torchvision.transforms as T
from data.transforms.base_transform import BaseTransform


class TorchvisionWrapper(BaseTransform):
    """Generic deterministic wrapper for torchvision transforms.
    
    This wrapper can wrap any torchvision transform to ensure deterministic behavior
    by properly handling the seed parameter and controlling PyTorch's global random state.
    
    According to Pylon's design philosophy (CLAUDE.md Section 3.1):
    - NO DEFENSIVE PROGRAMMING - FAIL FAST AND LOUD
    - Use assertions to enforce contracts through input validation
    - Let code crash with clear error messages if assumptions are violated
    """
    
    def __init__(self, transform_class: Type, **kwargs: Any) -> None:
        """Initialize TorchvisionWrapper with any torchvision transform.
        
        Args:
            transform_class: The torchvision transform class to wrap
            **kwargs: Arguments to pass to the transform class constructor
            
        Raises:
            AssertionError: If transform_class is not a valid torchvision transform
        """
        # Input validation with assertions as per CLAUDE.md guidelines
        assert transform_class is not None, "transform_class must not be None"
        assert hasattr(transform_class, '__module__'), f"transform_class must have __module__ attribute, got {type(transform_class)}"
        assert 'torchvision' in transform_class.__module__, f"transform_class must be from torchvision, got module: {transform_class.__module__}"
        assert callable(transform_class), f"transform_class must be callable, got {type(transform_class)}"
        
        self.transform_class = transform_class
        self.transform_kwargs = kwargs
        
        # Create the underlying torchvision transform
        self._transform = transform_class(**kwargs)
    
    def _call_single(self, image: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Apply torchvision transform with deterministic seeding.
        
        Args:
            image: Input tensor image
            generator: Optional generator for deterministic behavior
            
        Returns:
            Transformed image tensor
            
        Raises:
            AssertionError: If input validation fails
        """
        # Input validation as per CLAUDE.md guidelines
        assert isinstance(image, torch.Tensor), f"image must be torch.Tensor, got {type(image)}"
        assert image.numel() > 0, "image tensor must not be empty"
        
        if generator is not None:
            # Save current global random state
            current_state = torch.get_rng_state()
            
            try:
                # Use the generator's initial seed for consistent behavior
                # This ensures the same seed always produces the same result
                seed = generator.initial_seed()
                torch.manual_seed(seed)
                
                # Apply torchvision transform with controlled random state
                result = self._transform(image)
            finally:
                # Always restore original global state
                torch.set_rng_state(current_state)
            
            return result
        else:
            # No generator provided, use default behavior
            return self._transform(image)
    
    def __str__(self) -> str:
        """String representation showing only the inner transform."""
        transform_name = self.transform_class.__name__
        formatted_params = self.format_params(self.transform_kwargs)
        
        if formatted_params:
            return f"{transform_name}({formatted_params})"
        else:
            return f"{transform_name}()"