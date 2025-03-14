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
        
    def get_transform(self, index: int) -> Optional[Callable]:
        """Get a registered transform function.
        
        Args:
            index: Index of the transform
            
        Returns:
            Transform function or None if not found
        """
        try:
            return self._transforms[index]
        except IndexError:
            return None
        
    def apply_transforms(self, data: Any, transform_indices: List[int]) -> Optional[Any]:
        """Apply a sequence of transforms to data.
        
        Args:
            data: Data to transform
            transform_indices: List of transform indices to apply
            
        Returns:
            Transformed data or None if any transform fails
        """
        result = data
        for idx in transform_indices:
            transform = self.get_transform(idx)
            if transform is None:
                self.logger.error(f"Transform at index {idx} not found")
                return None
            try:
                result = transform(result)
            except Exception as e:
                self.logger.error(f"Failed to apply transform {idx}: {str(e)}")
                return None
        return result
        
    def build_transform_compose(self, transform_indices: List[int]) -> Optional[Compose]:
        """Build a Compose transform from a list of transform indices.
        
        Args:
            transform_indices: List of transform indices to compose
            
        Returns:
            Compose transform object or None if any transform is not found
        """
        transforms: List[Tuple[Callable, Optional[Dict[str, Any]]]] = []
        
        for idx in transform_indices:
            transform = self.get_transform(idx)
            if transform is None:
                self.logger.error(f"Transform at index {idx} not found")
                return None
            transforms.append((transform, None))  # None for default params
            
        return Compose(transforms=transforms)
        
    def clear_transforms(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()
        
    def get_num_transforms(self) -> int:
        """Get number of registered transforms.
        
        Returns:
            Number of transforms
        """
        return len(self._transforms)
        
    def register_transforms_from_config(self, transforms_cfg: Dict[str, Any]) -> None:
        """Register transforms from a configuration dictionary.
        
        Args:
            transforms_cfg: Transform configuration dictionary
        """
        assert isinstance(transforms_cfg, dict), "Transform configuration must be a dictionary"
        assert 'class' in transforms_cfg, "Transform configuration must contain 'class' key"
        assert 'args' in transforms_cfg, "Transform configuration must contain 'args' key"

        transforms = transforms_cfg['args'].get('transforms', [])
        for transform, _ in transforms:
            self.register_transform(transform)
