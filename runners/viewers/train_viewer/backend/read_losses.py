from typing import List
import os

import torch


def read_losses(logs_dirpath: str) -> List[torch.Tensor]:
    """Read training losses from log directory.

    Reads consecutively from epoch_0, epoch_1, etc. until first unavailable file.

    Args:
        logs_dirpath: Path to logs directory

    Returns:
        List of loss tensors, one per epoch
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert logs_dirpath is not None, "logs_dirpath must not be None"
    assert isinstance(logs_dirpath, str), f"logs_dirpath must be str, got {type(logs_dirpath)}"
    assert os.path.isdir(logs_dirpath), f"logs_dirpath must exist and be path to a directory: {logs_dirpath}"

    results = []
    expected_length = None
    idx = 0

    while True:
        epoch_dir = os.path.join(logs_dirpath, f'epoch_{idx}')
        losses_file = os.path.join(epoch_dir, 'training_losses.pt')

        # Stop when we reach first unavailable file
        if not os.path.exists(losses_file):
            break

        losses = torch.load(losses_file)

        # Validate that losses is a 1D torch tensor
        assert isinstance(losses, torch.Tensor), f"Loaded losses must be torch.Tensor, got {type(losses)}"
        assert losses.dim() == 1, f"Losses tensor must be 1D, got {losses.dim()}D tensor with shape {losses.shape}"

        # Validate that all epochs have the same length
        if expected_length is None:
            expected_length = len(losses)
        else:
            assert len(losses) == expected_length, f"All epochs must have same length. Expected {expected_length}, got {len(losses)} for epoch {idx}"

        results.append(losses)
        idx += 1

    assert len(results) > 0, f"No epoch directories with training_losses.pt found in {logs_dirpath}"

    return results
