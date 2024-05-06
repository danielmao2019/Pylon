from typing import List, Optional
import torch

from ._base_ import GradientManipulationBaseOptimizer
import utils


class AlignedMTLOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Independent Component Alignment for Multi-Task Learning (https://arxiv.org/pdf/2305.19000.pdf)
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
        # compute Gram matrix
        gram_matrix = utils.gradients.get_gram_matrix(grads_list)
        # compute eigenvalues and eigenvectors
        L, V = torch.linalg.eig(gram_matrix)
        assert L.dtype == V.dtype == torch.complex64, f"{L.dtype=}, {V.dtype=}"
        assert torch.all(L.imag == 0) and torch.all(V.imag == 0)
        L = L.real
        V = V.real
        # compute balance matrix
        sigma_min = L.min().sqrt()
        balance_matrix = sigma_min * torch.matmul(V, torch.matmul(torch.diag(1/L.sqrt()), V.t()))
        # compute final gradient
        alpha = balance_matrix.mean(dim=1)
        assert alpha.shape == (self.num_tasks,), f"{alpha.shape=}"
        result = sum([grads_list[i] * alpha[i] for i in range(self.num_tasks)])
        return result
