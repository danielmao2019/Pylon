from typing import List, Optional
import random
import torch

from ._base_ import GradientManipulationBaseOptimizer


class PCGradOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Gradient Surgery for Multi-Task Learning (https://arxiv.org/abs/2001.06782)

    References
        * https://github.com/WeiChengTseng/Pytorch-PCGrad (Sun May 28, 2023)
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
        result = torch.zeros_like(grads_list[0])
        for i in range(len(grads_list)):
            gi = grads_list[i].detach().clone()
            idx_j_order = list(range(len(grads_list)))
            idx_j_order.remove(i)
            random.shuffle(idx_j_order)
            for j in idx_j_order:
                gj = grads_list[j]
                inner_product = torch.dot(gi, gj)
                if inner_product < 0:
                    gi -= (inner_product / (torch.linalg.vector_norm(gj)**2)) * gj
            result += gi
        result /= len(grads_list)
        return result
