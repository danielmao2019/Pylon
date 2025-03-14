"""Transform manager module."""
from typing import Dict, Any, List, Callable, Optional, Tuple
import logging
from data.transforms.compose import Compose


class TransformManager:
    """Manages dataset transformations."""

    def __init__(self):
        """Initialize the transform manager."""
        self.logger = logging.getLogger(__name__)
        self._transforms: List[Callable] = []
        
    def register_transform(self, transform_fn: Callable) -> None:
        """Register a transform function.
        
        Args:
            transform_fn: Transform function to register
        """
        self._transforms.append(transform_fn)
        
    def get_transform_info(self, index: int) -> Dict[str, Any]:
        """Get information about a transform.
        
        Args:
            index: Index of the transform
            
        Returns:
            Dictionary containing transform information
        """
        assert 0 <= index < len(self._transforms), f"Transform index {index} out of range [0, {len(self._transforms)})"
        transform = self._transforms[index]
        return {
            'index': index,
            'name': transform.__class__.__name__
        }
        
    def get_available_transforms(self) -> List[Dict[str, Any]]:
        """Get information about all available transforms.
        
        Returns:
            List of dictionaries containing transform information
        """
        return [
            self.get_transform_info(i) for i in range(len(self._transforms))
        ]
        
    def apply_transforms(self, data: Any, transform_indices: List[int]) -> Optional[Any]:
        """Apply a sequence of transforms to data.
        
        Args:
            data: Data to transform
            transform_indices: List of transform indices to apply
            
        Returns:
            Transformed data or None if any transform fails
        """
        transforms: List[Tuple[Callable, Optional[Dict[str, Any]]]] = []
        
        for idx in transform_indices:
            assert 0 <= idx < len(self._transforms), f"Transform index {idx} out of range [0, {len(self._transforms)})"
            transforms.append((self._transforms[idx], None))  # None for default params
            
        compose = Compose(transforms=transforms)
        return compose(data)

    def clear_transforms(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()

    def register_transforms_from_config(self, transforms_cfg: Dict[str, Any]) -> None:
        """Register transforms from a configuration dictionary.
        
        Args:
            transforms_cfg: Transform configuration dictionary
        """
        assert isinstance(transforms_cfg, dict), f"Transform configuration must be a dictionary. Got {type(transforms_cfg)}."
        assert 'class' in transforms_cfg, f"Transform configuration must contain 'class' key. Got {transforms_cfg.keys()}."
        assert 'args' in transforms_cfg, f"Transform configuration must contain 'args' key. Got {transforms_cfg.keys()}."

        transforms = transforms_cfg['args'].get('transforms', [])
        for transform, _ in transforms:
            self.register_transform(transform)
