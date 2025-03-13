"""Callback registry for managing viewer callbacks.

This module provides a decorator-based callback registration system for the viewer.
It allows for better organization and dependency management of callbacks.
"""
from typing import Dict, Any, List, Callable, Optional, TypeVar, Union
from functools import wraps
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

# Type variables for generic typing
T = TypeVar('T')
CallbackInput = Union[Input, State]
CallbackOutput = Output


class CallbackRegistry:
    """Registry for managing viewer callbacks.

    This class provides a centralized registry for managing callbacks in the viewer.
    It supports:
    - Decorator-based callback registration
    - Dependency management between callbacks
    - Type hints for callback inputs/outputs
    - Callback grouping and organization
    """

    def __init__(self):
        """Initialize the callback registry."""
        self._callbacks: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}

    def callback(self, outputs: Union[CallbackOutput, List[CallbackOutput]],
                inputs: List[CallbackInput],
                states: Optional[List[CallbackInput]] = None,
                group: str = "default"):
        """Decorator for registering callbacks.

        Args:
            outputs: Output(s) of the callback
            inputs: Input(s) of the callback
            states: Optional state(s) of the callback
            group: Group name for organizing callbacks

        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            # Convert single output to list
            if isinstance(outputs, CallbackOutput):
                outputs_list = [outputs]
            else:
                outputs_list = outputs

            # Create callback ID
            callback_id = f"{group}.{func.__name__}"

            # Store callback info
            self._callbacks[callback_id] = {
                'function': func,
                'outputs': outputs_list,
                'inputs': inputs,
                'states': states or [],
                'group': group
            }

            # Add to dependencies
            if group not in self._dependencies:
                self._dependencies[group] = []
            self._dependencies[group].append(callback_id)

            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except PreventUpdate:
                    raise
                except Exception as e:
                    print(f"Error in callback {callback_id}: {e}")
                    raise

            return wrapper

        return decorator

    def register_callbacks(self, app: Any) -> None:
        """Register all callbacks with the Dash app.

        Args:
            app: Dash application instance
        """
        for callback_id, callback_info in self._callbacks.items():
            # Get outputs and check if any have allow_duplicate=True
            outputs = callback_info['outputs']
            has_duplicate = any(
                isinstance(output, dict) and output.get('allow_duplicate')
                for output in (outputs if isinstance(outputs, list) else [outputs])
            )
            
            # If any output allows duplicates, add prevent_initial_call
            kwargs = {}
            if has_duplicate:
                kwargs['prevent_initial_call'] = True
            
            app.callback(
                callback_info['outputs'],
                callback_info['inputs'],
                callback_info['states'],
                **kwargs
            )(callback_info['function'])

    def get_callbacks(self, group: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get all callbacks or callbacks for a specific group.

        Args:
            group: Optional group name to filter by

        Returns:
            Dictionary of callbacks
        """
        if group is None:
            return self._callbacks
        return {k: v for k, v in self._callbacks.items() if v['group'] == group}

    def get_dependencies(self, group: Optional[str] = None) -> Dict[str, List[str]]:
        """Get callback dependencies.

        Args:
            group: Optional group name to filter by

        Returns:
            Dictionary of dependencies
        """
        if group is None:
            return self._dependencies
        return {k: v for k, v in self._dependencies.items() if k == group}

    def clear(self) -> None:
        """Clear all registered callbacks."""
        self._callbacks.clear()
        self._dependencies.clear()


# Create global registry instance
registry = CallbackRegistry()


def callback(outputs: Union[CallbackOutput, List[CallbackOutput]],
            inputs: List[CallbackInput],
            states: Optional[List[CallbackInput]] = None,
            group: str = "default"):
    """Decorator for registering callbacks.

    This is a convenience wrapper around the registry's callback decorator.

    Args:
        outputs: Output(s) of the callback
        inputs: Input(s) of the callback
        states: Optional state(s) of the callback
        group: Group name for organizing callbacks

    Returns:
        Decorator function
    """
    return registry.callback(outputs, inputs, states, group)
