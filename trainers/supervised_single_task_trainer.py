import torch
from .base_trainer import BaseTrainer


class SupervisedSingleTaskTrainer(BaseTrainer):
    __doc__ = r"""Trainer class for supervised single-task learning.
    """

    def _set_gradients_(self, example: dict):
        r"""Default method to set gradients.
        """
        self.optimizer.zero_grad()
        assert 'losses' in example
        losses = example['losses']
        if type(losses) == dict:
            losses = torch.stack(list(losses.values()), dim=0)
            losses = losses.sum()
        else:
            assert type(losses) == torch.Tensor
        assert losses.numel() == 1
        losses.backward()
