from typing import Optional

import torch


def get_layer_by_name(model: torch.nn.Module, layer_name: str) -> Optional[torch.nn.Module]:
    """Get a layer from the model by its name/path.

    Args:
        model: The model
        layer_name: Dot-separated path to the layer (e.g., 'backbone.layer4.1.conv2')

    Returns:
        The target layer or None if not found
    """
    try:
        layer = model
        for part in layer_name.split('.'):
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                return None
        return layer
    except Exception:
        return None
