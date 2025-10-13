"""Deterministic hashing utilities for Python objects."""

from typing import Any, Union
import hashlib


def deterministic_hash(obj: Any) -> int:
    """Create a deterministic hash for any Python object.

    This function provides consistent, deterministic hashing that works across
    different Python sessions (unlike built-in hash() which is randomized).

    Supports common Python types:
    - Primitives: int, float, str, bool, None
    - Collections: list, tuple, set, frozenset, dict
    - Nested structures of the above types

    Args:
        obj: Any Python object to hash

    Returns:
        Deterministic integer hash value in range [0, 2^32)

    Examples:
        >>> deterministic_hash("hello")
        1564557354
        >>> deterministic_hash((1, 2, 3))
        1107277306
        >>> deterministic_hash({"a": 1, "b": [2, 3]})
        2849572891
    """
    hasher = hashlib.md5()
    _hash_recursive(obj, hasher)
    # Take first 8 hex chars and convert to int, then mod to ensure 32-bit
    return int(hasher.hexdigest()[:8], 16) % (2**32)


def _hash_recursive(obj: Any, hasher: hashlib.md5) -> None:
    """Recursively hash an object into the given hasher.

    This function handles the recursive traversal and type-specific hashing
    logic to ensure deterministic, collision-resistant hashing.

    Args:
        obj: Object to hash
        hasher: MD5 hasher to update with object data
    """
    # Include type information to distinguish between different types
    # that might have the same string representation
    obj_type = type(obj).__name__
    hasher.update(f"<{obj_type}>".encode('utf-8'))

    if obj is None:
        hasher.update(b"None")

    elif isinstance(obj, bool):
        # Handle bool before int since bool is subclass of int
        hasher.update(str(obj).encode('utf-8'))

    elif isinstance(obj, (int, float)):
        hasher.update(str(obj).encode('utf-8'))

    elif isinstance(obj, str):
        hasher.update(obj.encode('utf-8'))

    elif isinstance(obj, bytes):
        hasher.update(obj)

    elif isinstance(obj, (list, tuple)):
        hasher.update(f"len:{len(obj)}".encode('utf-8'))
        for i, item in enumerate(obj):
            hasher.update(f"[{i}]".encode('utf-8'))
            _hash_recursive(item, hasher)

    elif isinstance(obj, (set, frozenset)):
        # Sort set elements by their hash to ensure consistent ordering
        sorted_items = sorted(obj, key=lambda x: deterministic_hash(x))
        hasher.update(f"len:{len(obj)}".encode('utf-8'))
        for i, item in enumerate(sorted_items):
            hasher.update(f"[{i}]".encode('utf-8'))
            _hash_recursive(item, hasher)

    elif isinstance(obj, dict):
        # Sort by keys to ensure consistent ordering
        sorted_keys = sorted(obj.keys(), key=lambda x: deterministic_hash(x))
        hasher.update(f"len:{len(obj)}".encode('utf-8'))
        for key in sorted_keys:
            hasher.update("key:".encode('utf-8'))
            _hash_recursive(key, hasher)
            hasher.update("val:".encode('utf-8'))
            _hash_recursive(obj[key], hasher)

    else:
        # Fallback for other types: use string representation
        # Include class name and module for better uniqueness
        fallback_repr = (
            f"{obj.__class__.__module__}.{obj.__class__.__name__}:{str(obj)}"
        )
        hasher.update(fallback_repr.encode('utf-8'))


def convert_to_seed(obj: Any) -> int:
    """Convert any object to a deterministic integer seed.

    This is a convenience wrapper around deterministic_hash() that's specifically
    designed for use as random number generator seeds.

    Args:
        obj: Any object to convert to a seed

    Returns:
        Integer seed suitable for random number generators

    Examples:
        >>> convert_to_seed("hello")
        1564557354
        >>> convert_to_seed((1, 2, 3))
        1107277306
    """
    if isinstance(obj, int):
        # If already an int, just ensure it's in valid range
        return obj % (2**32)
    else:
        return deterministic_hash(obj)
