from typing import Any, Optional
import os
import json
import threading
import jsbeautifier
from datetime import datetime
from dataclasses import is_dataclass, asdict
from utils.ops.apply import apply_tensor_op, apply_op
import torch
import numpy as np

# Global lock registry for thread-safe JSON file operations
_json_file_locks = {}
_json_locks_lock = threading.Lock()


def _get_json_file_lock(filepath: str) -> threading.Lock:
    """Get or create a lock for the specific JSON file."""
    abs_path = os.path.abspath(filepath)
    with _json_locks_lock:
        if abs_path not in _json_file_locks:
            _json_file_locks[abs_path] = threading.Lock()
        return _json_file_locks[abs_path]


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


def load_json(filepath: str) -> Any:
    """Load JSON from file.
    
    Args:
        filepath: Path to JSON file to load
        
    Returns:
        Loaded JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def safe_load_json(filepath: str) -> Any:
    """Thread-safe JSON loading with file locking.
    
    Args:
        filepath: Path to JSON file to load
        
    Returns:
        Loaded JSON data
        
    Raises:
        AssertionError: If file doesn't exist or is empty
        Exception: Any other error with filepath context
    """
    file_lock = _get_json_file_lock(filepath)
    try:
        with file_lock:
            # Input validation inside lock
            assert os.path.exists(filepath), f"File does not exist: {filepath}"
            assert os.path.getsize(filepath) > 0, f"File is empty: {filepath}"
            
            return load_json(filepath)
    except Exception as e:
        # Re-raise with filepath context for all errors
        raise type(e)(f"Error loading JSON from {filepath}: {e}") from e


def safe_save_json(obj: Any, filepath: str) -> None:
    """Thread-safe JSON saving with file locking and automatic serialization.
    
    Automatically handles dataclasses, torch.Tensor, numpy.ndarray, 
    datetime, and all nested data structures without requiring manual conversion.
    
    Args:
        obj: Object to save (dataclasses are automatically converted)
        filepath: Path to save JSON file
        
    Raises:
        AssertionError: If directory doesn't exist
        Exception: Any other error with filepath context
    """
    file_lock = _get_json_file_lock(filepath)
    try:
        with file_lock:
            # Input validation inside lock
            assert (
                os.path.dirname(filepath) == "" or
                os.path.isdir(os.path.dirname(filepath))
            ), f"Directory does not exist for file: {filepath}, directory: {os.path.dirname(filepath)}"
            
            save_json(obj, filepath)
    except Exception as e:
        # Re-raise with filepath context for all errors
        raise type(e)(f"Error saving JSON to {filepath}: {e}") from e


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
