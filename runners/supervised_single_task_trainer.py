from typing import Dict, Any
import torch
from .base_trainer import BaseTrainer


class SupervisedSingleTaskTrainer(BaseTrainer):
    __doc__ = r"""Trainer class for supervised single-task learning.
    """

    def _set_gradients_(self, dp: Dict[str, Dict[str, Any]]) -> None:
        r"""Set gradients in single-task learning setting.
        """
        self.optimizer.zero_grad()
        assert 'losses' in dp
        losses = dp['losses']
        if type(losses) == dict:
            losses = torch.stack(list(losses.values()), dim=0).sum()
        assert type(losses) == torch.Tensor, f"{type(losses)=}"
        assert losses.numel() == 1, f"{losses.shape=}"
        losses.backward()
