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

    def get_transform_info(self, index: int) -> Optional[Dict[str, Any]]:
        """Get information about a transform.
        
        Args:
            index: Index of the transform
            
        Returns:
            Dictionary containing transform information or None if not found
        """
        transform = self.get_transform(index)
        if transform is None:
            return None
        return {
            'index': index,
            'name': transform.__class__.__name__,
            'description': transform.__doc__ or "No description available"
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
        assert isinstance(transforms_cfg, dict), f"Transform configuration must be a dictionary. Got {type(transforms_cfg)}."
        assert 'class' in transforms_cfg, f"Transform configuration must contain 'class' key. Got {transforms_cfg.keys()}."
        assert 'args' in transforms_cfg, f"Transform configuration must contain 'args' key. Got {transforms_cfg.keys()}."

        transforms = transforms_cfg['args'].get('transforms', [])
        for transform, _ in transforms:
            self.register_transform(transform)
