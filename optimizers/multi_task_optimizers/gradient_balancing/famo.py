from typing import Tuple, Dict, Union
import torch

from ._base_ import GradientBalancingBaseOptimizer


class FAMOOptimizer(GradientBalancingBaseOptimizer):
    __doc__ = r"""
    Paper: [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792).
    """

    def __init__(self, **kwargs) -> None:
        super(FAMOOptimizer, self).__init__(**kwargs)
        self.weights = None
        self.prev_losses = None
        self.first_iter = True
        self.epsilon = 1.0e-08

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # initialization
        beta = self.optimizer.param_groups[0]['lr']
        gamma = 0.001
        losses_tensor = torch.stack(list(losses.values()))
        # compute weights
        with torch.no_grad():
            if self.first_iter:
                self.weights = torch.zeros_like(losses_tensor)
                self.first_iter = False
            else:
                z = torch.nn.Softmax(dim=0)(self.weights)
                delta = torch.matmul(
                    torch.diag(z) - torch.outer(z, z),
                    torch.log(self.prev_losses + self.epsilon) - torch.log(losses_tensor + self.epsilon)
                )
                assert len(delta.shape) == 1, f"{delta.shape=}"
                self.weights = self.weights - beta * (delta + gamma * self.weights)
        # reweigh losses
        total_loss = self._reweigh_losses_(losses=losses_tensor)
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # update buffers
        self.prev_losses = losses_tensor.detach().clone()

    def _reweigh_losses_(
        self,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Returns:
            total_loss: the reweighed version of the losses based on self.weights.
        """
        assert self.weights.shape == losses.shape, f"{self.weights.shape=}, {losses.shape=}"
        with torch.no_grad():
            z = torch.nn.Softmax(dim=0)(self.weights)
            odiv = z / (losses + self.epsilon)
            v = odiv / odiv.sum()
        assert not v.requires_grad
        total_loss = self.num_tasks * (v * losses).sum()
        return total_loss
