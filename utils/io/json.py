from typing import Any
import os
import json
import tempfile
import jsbeautifier
from datetime import datetime
from dataclasses import is_dataclass, asdict
from utils.ops.apply import apply_tensor_op, apply_op
import torch
import numpy as np


def serialize_tensor(obj: Any) -> Any:
    """Serialize torch tensors to lists (backward compatibility)."""
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
    """Load JSON from file with error handling.
    
    Args:
        filepath: Path to JSON file to load
        
    Returns:
        Loaded JSON data
        
    Raises:
        RuntimeError: If file doesn't exist, is empty, or has invalid JSON
    """
    # Input validation
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File is empty: {filepath}")

    try:
        # Load JSON
        with open(filepath, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        # Re-raise with filepath context for all errors
        raise RuntimeError(f"Error loading JSON from {filepath}: {e}") from e


def save_json(obj: Any, filepath: str) -> None:
    """Save object to JSON file using atomic writes and automatic serialization.
    
    Uses atomic writes (temp file + rename) to prevent race conditions between processes
    and threads. The rename operation is atomic at the filesystem level, ensuring readers 
    never see partially written files.
    
    Automatically handles dataclasses, torch.Tensor, numpy.ndarray, datetime,
    and all nested data structures without requiring manual conversion.
    
    Args:
        obj: Object to save (dataclasses are automatically converted)
        filepath: Path to save JSON file
        
    Raises:
        RuntimeError: If directory doesn't exist or write operation fails
    """
    try:
        # Input validation
        target_dir = os.path.dirname(filepath)
        if target_dir and not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Directory does not exist: {target_dir}")
        
        # Atomic write using temp file + rename
        temp_fd = None
        temp_filepath = None
        
        try:
            # Create temp file in same directory as target file
            # (rename is only atomic within the same filesystem)
            temp_fd, temp_filepath = tempfile.mkstemp(
                suffix='.tmp', 
                prefix='json_', 
                dir=target_dir or '.'
            )
            
            # Close the file descriptor - we'll use our own file operations
            os.close(temp_fd)
            temp_fd = None
            
            # Serialize and write to temporary file
            serialized_obj = serialize_object(obj)
            with open(temp_filepath, 'w') as f:
                f.write(jsbeautifier.beautify(
                    json.dumps(serialized_obj), 
                    jsbeautifier.default_options()
                ))
            
            # Atomic rename - this prevents race conditions
            os.rename(temp_filepath, filepath)
            temp_filepath = None  # Success - no cleanup needed
            
        except Exception as e:
            # Cleanup temp file if something went wrong
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except:
                    pass
            if temp_filepath is not None and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            raise
            
    except Exception as e:
        # Re-raise with filepath context for all errors
        raise RuntimeError(f"Error saving JSON to {filepath}: {e}") from e


# Backward compatibility aliases for existing code
safe_load_json = load_json
safe_save_json = save_json