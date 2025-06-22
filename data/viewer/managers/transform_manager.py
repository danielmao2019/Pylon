"""Transform manager module."""
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
import logging
from data.transforms.base_transform import BaseTransform
from data.transforms.compose import Compose


class TransformManager:
    """Manages dataset transformations."""

    def __init__(self):
        """Initialize the transform manager."""
        self.logger = logging.getLogger(__name__)
        # Store transforms in the normalized format: Dict[str, Any] with keys 'op', 'input_names', 'output_names'
        self._transforms: List[Dict[str, Any]] = []

    def register_transform(self, transform: Union[
        Tuple[Callable, Union[Tuple[str, str], List[Tuple[str, str]]]],
        Dict[str, Union[Callable, Union[Tuple[str, str], List[Tuple[str, str]]]]]
    ]) -> None:
        """Register a transform.

        Args:
            transform: Transform in either tuple format (transform_fn, input_keys) for backward compatibility
                     or dictionary format with keys 'op', 'input_names', and optionally 'output_names'
        """
        # Use the normalization method from Compose class for code reuse
        normalized_transform = Compose.normalize_transforms_cfg(transform)
        self._transforms.append(normalized_transform)

    def register_transforms_from_config(self, transforms_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Register transforms from a configuration dictionary.

        Args:
            transforms_cfg: Transform configuration dictionary. If None or empty, no transforms will be registered.
        """
        # Clear existing transforms
        self.clear_transforms()

        # If no transforms config provided, return without registering any transforms
        if not transforms_cfg:
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

    def _extract_transform_display_info(self, transform_op: Union[Any, Dict[str, Any]]) -> Tuple[str, str]:
        """Extract display name and string representation from a transform operation.

        Args:
            transform_op: Transform operation (callable or dict config)

        Returns:
            Tuple of (transform_name, transform_string)
        """
        if callable(transform_op):
            # Direct callable - try to use its string representation
            transform_name = transform_op.__class__.__name__
            try:
                transform_string = str(transform_op)
            except Exception:
                transform_string = transform_name
        elif isinstance(transform_op, dict) and 'class' in transform_op:
            # Dictionary config with class and args
            transform_class = transform_op['class']
            transform_name = transform_class.__name__ if hasattr(transform_class, '__name__') else str(transform_class)

            # Format the dict config as a string using BaseTransform helper
            args = transform_op.get('args', {})
            formatted_params = BaseTransform.format_params(args)

            if formatted_params:
                transform_string = f"{transform_name}({formatted_params})"
            else:
                transform_string = transform_name
        else:
            # This should never happen if transforms are properly normalized
            assert False, f"Should not reach here. Unexpected transform_op type: {type(transform_op)}"

        return transform_name, transform_string

    def get_transform_info(self, index: int) -> Dict[str, Any]:
        """Get information about a transform.

        Args:
            index: Index of the transform

        Returns:
            Dictionary containing transform information
        """
        assert 0 <= index < len(self._transforms), f"Transform index {index} out of range [0, {len(self._transforms)})"
        transform = self._transforms[index]

        # Since transforms are normalized, we can directly access the keys
        transform_name, transform_string = self._extract_transform_display_info(transform['op'])

        return {
            'index': index,
            'name': transform_name,
            'string': transform_string,
            'input_names': transform['input_names'],
            'output_names': transform['output_names']
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
        # Select the transforms in the order specified by indices
        # Since our transforms are already normalized, we can use them directly
        selected_transforms = [self._transforms[idx] for idx in transform_indices]
        compose = Compose(transforms=[])  # Create empty compose
        compose.transforms = selected_transforms  # Directly assign normalized transforms
        return compose(data)
