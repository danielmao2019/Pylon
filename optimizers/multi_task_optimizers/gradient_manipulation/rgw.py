from typing import List, Optional
import torch

from ._base_ import GradientManipulationBaseOptimizer


class RGWOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (https://arxiv.org/abs/2111.10603)
    """

    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            grads_list (List[torch.Tensor]): the list of 1D gradient tensors of each objective.
            shared_rep (torch.Tensor): unused argument.
        Returns:
            result (torch.Tensor): the 1D manipulated gradient tensor.
        """
        # input checks
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        # compute result
        weights_list: torch.Tensor = torch.nn.Softmax(dim=0)(torch.normal(mean=0, std=1, size=(self.num_tasks,)))
        weights_list: List[float] = weights_list.tolist()
        result = sum([
            weight * grad for (weight, grad) in zip(weights_list, grads_list)
        ])
        return result
