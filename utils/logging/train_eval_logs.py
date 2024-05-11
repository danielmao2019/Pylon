from typing import Dict, Union, Optional
import torch
import wandb


def log_losses(
    losses: Union[torch.Tensor, Dict[str, torch.Tensor]],
    wandb_log: Optional[bool] = False,
) -> dict:
    r"""
    Args:
        losses (torch.Tensor or Dict[str, torch.Tensor]): either single loss or multiple losses.
    """
    if type(losses) == torch.Tensor:
        data = {'loss': losses}
    elif type(losses) == dict:
        data = {f"loss_{name}": losses[name] for name in losses}
    else:
        raise TypeError(f"[ERROR] Losses logging method only implemented for torch.Tensor and Dict[str, torch.Tensor]. Got {type(losses)}.")
    if wandb_log:
        wandb.log(data=data)
    return data


def log_scores(
    scores: Union[torch.Tensor, Dict[str, torch.Tensor]],
    wandb_log: Optional[bool] = False,
) -> dict:
    r"""
    Args:
        scores (torch.Tensor or Dict[str, torch.Tensor]): either single score or multiple scores.
    """
    if type(scores) == torch.Tensor:
        data = {'score': scores}
    elif type(scores) == dict:
        data = {f"score_{name}": scores[name] for name in scores}
    else:
        raise TypeError(f"[ERROR] Scores logging method only implemented for torch.Tensor and Dict[str, torch.Tensor]. Got {type(scores)}.")
    if wandb_log:
        wandb.log(data=data)
    return data
