from typing import Any, Dict
from abc import abstractmethod
import torch
from debugger.base_debugger import BaseDebugger


class ForwardDebugger(BaseDebugger):
    """Base class for debuggers that use PyTorch forward hooks."""

    def __init__(self, layer_name: str):
        """Initialize forward debugger.

        Args:
            layer_name: Dot-separated path to layer (e.g., 'backbone.layer4')
        """
        self.layer_name = layer_name
        self.last_capture = None

    def forward_hook_fn(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        """PyTorch forward hook function."""
        self.last_capture = self.process_forward(module, input, output)

    @abstractmethod
    def process_forward(self, module: torch.nn.Module, input: Any, output: Any) -> Any:
        """Process data from forward hook.

        Args:
            module: The layer this hook is attached to
            input: Input to the layer
            output: Output from the layer

        Returns:
            Processed debug data
        """
        raise NotImplementedError

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Any:
        """Return the captured data from forward pass."""
        return self.last_capture
