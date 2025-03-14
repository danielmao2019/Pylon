"""Transform manager module."""
from typing import Dict, Any, List, Callable, Optional
import logging

class TransformManager:
    """Manages dataset transformations."""

    def __init__(self):
        """Initialize the transform manager."""
        self.logger = logging.getLogger(__name__)
        self._transforms: Dict[str, Callable] = {}
        
    def register_transform(self, name: str, transform_fn: Callable) -> None:
        """Register a transform function.
        
        Args:
            name: Name of the transform
            transform_fn: Transform function to register
        """
        if name in self._transforms:
            self.logger.warning(f"Overwriting existing transform: {name}")
        self._transforms[name] = transform_fn
        
    def get_transform(self, name: str) -> Optional[Callable]:
        """Get a registered transform function.
        
        Args:
            name: Name of the transform
            
        Returns:
            Transform function or None if not found
        """
        return self._transforms.get(name)
        
    def apply_transform(self, name: str, data: Any) -> Optional[Any]:
        """Apply a transform to data.
        
        Args:
            name: Name of the transform to apply
            data: Data to transform
            
        Returns:
            Transformed data or None if transform fails
        """
        transform = self.get_transform(name)
        if transform is None:
            self.logger.error(f"Transform not found: {name}")
            return None
            
        try:
            return transform(data)
        except Exception as e:
            self.logger.error(f"Error applying transform {name}: {str(e)}")
            return None
            
    def clear_transforms(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()
        
    def get_transform_names(self) -> List[str]:
        """Get names of all registered transforms.
        
        Returns:
            List of transform names
        """
        return list(self._transforms.keys())
        
    def register_transforms_from_config(self, transforms_cfg: Dict[str, Any]) -> None:
        """Register transforms from a configuration dictionary.
        
        Args:
            transforms_cfg: Transform configuration dictionary
        """
        if not isinstance(transforms_cfg, dict):
            self.logger.error("Invalid transforms configuration")
            return
            
        if 'class' not in transforms_cfg or 'args' not in transforms_cfg:
            self.logger.error("Transform config must contain 'class' and 'args' keys")
            return
            
        transforms = transforms_cfg['args'].get('transforms', [])
        for i, (transform, _) in enumerate(transforms):
            self.register_transform(f"transform_{i}", transform) 