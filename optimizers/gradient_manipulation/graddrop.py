from typing import List, Optional
import torch

from ._base_ import GradientManipulationBaseOptimizer


class GradDropOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (https://arxiv.org/abs/2010.06808)
    """

    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            grads_list (List[torch.Tensor]): the list of 1D gradient tensors of each objective.
        Returns:
            result (torch.Tensor): the 1D manipulated gradient tensor.
        """
        # input checks
        assert type(grads_list) == list
        for idx, grads in enumerate(grads_list):
            assert type(grads) == torch.Tensor, f"{idx=}, {type(grads)=}"
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        # initialization
        dim = len(grads_list[0])
        # multiply by sign if provided
        if self.wrt_rep:
            assert shared_rep is not None
            assert type(shared_rep) == torch.Tensor, f"{type(shared_rep)=}"
            signed_rep = torch.sign(shared_rep)
            for grads in grads_list:
                grads *= signed_rep
        # compute gradient purity
        torch_sum = torch.zeros(size=(dim,), dtype=torch.float32, device=torch.device('cuda'))
        torch_sum_abs = torch.zeros(size=(dim,), dtype=torch.float32, device=torch.device('cuda'))
        for idx in range(self.num_tasks):
            torch_sum += grads_list[idx]
            torch_sum_abs += torch.abs(grads_list[idx])
        gradient_purity = 0.5 * (1 + torch_sum / (torch_sum_abs+1.0e-09))
        assert 0 <= torch.min(gradient_purity) <= torch.max(gradient_purity) <= 1, f"{torch.min(gradient_purity)=}, {torch.max(gradient_purity)=}"
        prob = gradient_purity > torch.rand(size=gradient_purity.shape, device=torch.device('cuda'))
        result = torch.zeros(size=(dim,), dtype=torch.float32, device=torch.device('cuda'))
        for idx in range(self.num_tasks):
            mask = prob * (grads_list[idx] > 0) + (~prob) * (grads_list[idx] < 0)
            result += mask * grads_list[idx]
        return result
