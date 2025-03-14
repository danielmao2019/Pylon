"""Transform manager module."""
from typing import Dict, Any, List, Callable, Optional, Tuple
import logging
from data.transforms.compose import Compose


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
        
    def build_transform_compose(self, transform_names: List[str]) -> Optional[Compose]:
        """Build a Compose transform from a list of transform names.
        
        Args:
            transform_names: List of transform names to compose
            
        Returns:
            Compose transform object or None if any transform is not found
        """
        transforms: List[Tuple[Callable, Optional[Dict[str, Any]]]] = []
        
        for name in transform_names:
            transform = self.get_transform(name)
            if transform is None:
                self.logger.error(f"Transform not found: {name}")
                return None
            transforms.append((transform, None))  # None for default params
            
        return Compose(transforms=transforms)
        
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
        # Handle empty or None config
        if not transforms_cfg:
            self.logger.debug("No transforms configuration provided")
            return
            
        # Handle Compose object case
        if not isinstance(transforms_cfg, dict):
            transforms = getattr(transforms_cfg, 'transforms', [])
            for i, (transform, _) in enumerate(transforms):
                self.register_transform(f"transform_{i}", transform)
            return
            
        # Handle dictionary case
        if 'class' not in transforms_cfg or 'args' not in transforms_cfg:
            self.logger.debug("Transform config missing required keys")
            return
            
        transforms = transforms_cfg['args'].get('transforms', [])
        for i, (transform, _) in enumerate(transforms):
            self.register_transform(f"transform_{i}", transform)
