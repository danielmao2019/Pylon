from typing import Tuple, Dict, Union
import torch

from ._base_ import GradientBalancingBaseOptimizer


class RLWOptimizer(GradientBalancingBaseOptimizer):
    __doc__ = r"""Paper: Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/abs/2111.10603)
    """

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert len(losses) == self.num_tasks, f"{len(losses)=}, {self.num_tasks=}"
        # initialization
        losses_tensor = torch.stack(list(losses.values()))
        # compute weights
        with torch.no_grad():
            weights = torch.nn.Softmax(dim=0)(torch.normal(mean=0, std=1, size=(self.num_tasks,), device=torch.device('cuda')))
        assert not weights.requires_grad
        # reweigh losses
        total_loss = (weights * losses_tensor).sum()
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
