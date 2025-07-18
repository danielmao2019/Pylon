from typing import Any
import os
import json
import jsbeautifier
from datetime import datetime
from dataclasses import is_dataclass, asdict
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
    - dataclass -> dict (via asdict())
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
        
        # Handle dataclass objects
        elif is_dataclass(item):
            # Recursively serialize the dict representation
            return serialize_object(asdict(item))
        
        # All other types pass through unchanged (assumes JSON-serializable)
        else:
            return item
    
    return apply_op(func=_serialize_item, inputs=obj)


def save_json(obj: Any, filepath: str) -> None:
    """Save object to JSON file with automatic serialization.
    
    Automatically handles dataclasses, torch.Tensor, numpy.ndarray, 
    datetime, and all nested data structures without requiring manual conversion.
    
    Args:
        obj: Object to save (dataclasses are automatically converted)
        filepath: Path to save JSON file
    """
    assert (
        os.path.dirname(filepath) == "" or
        os.path.isdir(os.path.dirname(filepath))
    ), f"{filepath=}, {os.path.dirname(filepath)=}"
    
    # Automatically serialize the object (handles dataclasses, tensors, etc.)
    serialized_obj = serialize_object(obj)
        
    with open(filepath, mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(serialized_obj), jsbeautifier.default_options()))
