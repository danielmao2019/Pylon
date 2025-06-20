"""Transform manager module."""
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
import logging
from data.transforms.compose import Compose


class TransformManager:
    """Manages dataset transformations."""

    def __init__(self):
        """Initialize the transform manager."""
        self.logger = logging.getLogger(__name__)
        # Store transforms in the exact format expected by Compose
        self._transforms: List[Tuple[Callable, List[Tuple[str, str]]]] = []

    def register_transform(self, transform: Tuple[Callable, Union[Tuple[str, str], List[Tuple[str, str]]]]) -> None:
        """Register a transform.

        Args:
            transform: Tuple of (transform_fn, input_keys) where input_keys is either a single key pair
                     or a list of key pairs specifying which fields to transform
        """
        transform_fn, input_keys = transform
        # Convert single key pair to list if needed
        if isinstance(input_keys, tuple):
            input_keys = [input_keys]
        # Validate input keys
        assert isinstance(input_keys, list), f"input_keys must be a list, got {type(input_keys)}"
        for key_pair in input_keys:
            assert isinstance(key_pair, tuple), f"Each key pair must be a tuple, got {type(key_pair)}"
            assert len(key_pair) == 2, f"Each key pair must have length 2, got {len(key_pair)}"
            assert all(isinstance(k, str) for k in key_pair), f"Keys must be strings, got {key_pair}"
        self._transforms.append((transform_fn, input_keys))

    def register_transforms_from_config(self, transforms_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Register transforms from a configuration dictionary.

        Args:
            transforms_cfg: Transform configuration dictionary. If None or empty, no transforms will be registered.
        """
        # Clear existing transforms
        self.clear_transforms()

        # If no transforms config provided, return without registering any transforms
        if transforms_cfg is None:
            return

        # Validate config structure
        assert isinstance(transforms_cfg, dict), f"Transform configuration must be a dictionary. Got {type(transforms_cfg)}."
        assert 'class' in transforms_cfg, f"Transform configuration must contain 'class' key. Got {transforms_cfg.keys()}."
        assert 'args' in transforms_cfg, f"Transform configuration must contain 'args' key. Got {transforms_cfg.keys()}."

        # Register each transform directly from the config
        transforms = transforms_cfg['args'].get('transforms', [])
        for transform in transforms:
            self.register_transform(transform)

    def clear_transforms(self) -> None:
        """Clear all registered transforms."""
        self._transforms.clear()

    def get_transform_info(self, index: int) -> Dict[str, Any]:
        """Get information about a transform.

        Args:
            index: Index of the transform

        Returns:
            Dictionary containing transform information
        """
        assert 0 <= index < len(self._transforms), f"Transform index {index} out of range [0, {len(self._transforms)})"
        transform_fn, input_keys = self._transforms[index]
        return {
            'index': index,
            'name': transform_fn.__class__.__name__,
            'input_keys': input_keys
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
        # Simply select the transforms in the order specified by indices
        transforms = [self._transforms[idx] for idx in transform_indices]
        compose = Compose(transforms=transforms)
        return compose(data)
