from typing import Dict, Union
import torch


def log_losses(losses: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> dict:
    r"""
    Args:
        losses (torch.Tensor or Dict[str, torch.Tensor]): either single loss or multiple losses.
    """
    if type(losses) == torch.Tensor:
        return {'loss': losses}
    if type(losses) == dict:
        return {f"loss_{name}": losses[name] for name in losses}
    raise TypeError(f"[ERROR] Losses logging method only implemented for torch.Tensor and Dict[str, torch.Tensor]. Got {type(losses)}.")


def log_scores(scores: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> dict:
    r"""
    Args:
        scores (torch.Tensor or Dict[str, torch.Tensor]): either single score or multiple scores.
    """
    if type(scores) == torch.Tensor:
        return {'score': scores}
    if type(scores) == dict:
        return {f"score_{name}": scores[name] for name in scores}
    raise TypeError(f"[ERROR] Scores logging method only implemented for torch.Tensor and Dict[str, torch.Tensor]. Got {type(scores)}.")
