from typing import Any
import os
import json
import jsbeautifier
from datetime import datetime
from utils.ops.apply import apply_tensor_op, apply_op
import torch
import numpy as np


def serialize_tensor(obj: Any) -> Any:
    return apply_tensor_op(func=lambda x: x.detach().tolist(), inputs=obj)


def serialize_object(obj: Any) -> Any:
    """Serialize any nested object containing various data types to JSON-compatible format.
    
    Handles:
    - torch.Tensor -> list (via .detach().tolist())
    - numpy.ndarray -> list (via .tolist())
    - datetime -> ISO format string
    - NamedTuple -> dict (via ._asdict())
    - All other types -> unchanged (assumes JSON-serializable)
    
    Args:
        obj: Object to serialize (can be nested in tuples, lists, dicts)
        
    Returns:
        JSON-serializable version of the object
    """
    def _serialize_item(item: Any) -> Any:
        # Handle torch tensors
        if isinstance(item, torch.Tensor):
            return item.detach().tolist()
        
        # Handle numpy arrays
        elif isinstance(item, np.ndarray):
            return item.tolist()
        
        # Handle datetime objects
        elif isinstance(item, datetime):
            return item.isoformat()
        
        # Handle NamedTuple objects specifically (they have _fields attribute)
        elif hasattr(item, '_asdict') and hasattr(item, '_fields') and callable(getattr(item, '_asdict')):
            # Recursively serialize the dict representation
            return serialize_object(item._asdict())
        
        # All other types pass through unchanged (assumes JSON-serializable)
        else:
            return item
    
    return apply_op(func=_serialize_item, inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    """Save object to JSON file with generic serialization.
    
    Uses serialize_object which handles torch.Tensor, numpy.ndarray, 
    datetime, NamedTuple, and all nested data structures.
    
    Args:
        obj: Object to save
        filepath: Path to save JSON file
    """
    assert (
        os.path.dirname(filepath) == "" or
        os.path.isdir(os.path.dirname(filepath))
    ), f"{filepath=}, {os.path.dirname(filepath)=}"
    
    obj = serialize_object(obj)
        
    with open(filepath, mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(obj), jsbeautifier.default_options()))
