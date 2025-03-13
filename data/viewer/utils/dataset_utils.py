"""Utility functions for the dataset viewer."""
import os
import importlib.util
import json
import random
import numpy as np
import torch
import traceback
from pathlib import Path

def format_value(value):
    """Format a value for display in error messages."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, indent=2, default=str)
        except:
            return str(value)
    elif isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    else:
        return str(value)
