"""PyTorch tensor I/O utilities with atomic writes and automatic serialization."""

from typing import Any, Optional
import os
import torch
import tempfile


def load_torch(filepath: str, map_location: Optional[str] = None) -> Any:
    """Load PyTorch tensors/objects from file with error handling.

    Args:
        filepath: Path to .pt file to load
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda', 'cuda:0')

    Returns:
        Loaded PyTorch object

    Raises:
        RuntimeError: If file doesn't exist or loading fails
    """
    try:
        # Input validation
        assert os.path.isfile(
            filepath
        ), f"Path does not exist or is not a file: {filepath}"
        assert os.path.getsize(filepath) > 0, f"File is empty: {filepath}"

        # Load with map_location if specified
        if map_location is not None:
            return torch.load(filepath, map_location=map_location)
        else:
            return torch.load(filepath)

    except Exception as e:
        # Re-raise with filepath context for all errors
        raise RuntimeError(f"Error loading torch file from {filepath}: {e}") from e


def save_torch(obj: Any, filepath: str) -> None:
    """Save PyTorch object to file using atomic writes.

    Uses atomic writes (temp file + rename) to prevent race conditions between processes
    and threads. The rename operation is atomic at the filesystem level, ensuring readers
    never see partially written files.

    Args:
        obj: PyTorch object to save (tensors, dicts, models, etc.)
        filepath: Path to save .pt file

    Raises:
        RuntimeError: If directory doesn't exist or write operation fails
    """
    try:
        # Auto-create directory if it doesn't exist
        target_dir = os.path.dirname(filepath)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        # Atomic write using temp file + rename
        temp_fd = None
        temp_filepath = None

        try:
            # Create temp file in same directory as target file
            # (rename is only atomic within the same filesystem)
            temp_fd, temp_filepath = tempfile.mkstemp(
                suffix='.tmp', prefix='torch_', dir=target_dir or '.'
            )

            # Close the file descriptor - we'll use torch.save
            os.close(temp_fd)
            temp_fd = None

            # Save to temporary file
            torch.save(obj, temp_filepath)

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
        raise RuntimeError(f"Error saving torch file to {filepath}: {e}") from e
