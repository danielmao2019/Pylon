from typing import List, Union, Optional
import torch

from ._base_ import GradientManipulationBaseOptimizer
import utils


NUMERICAL_STABILITY = 1.0e-03


class MGDAOptimizer(GradientManipulationBaseOptimizer):

    def __init__(self, max_iter: Optional[int] = 25, **kwargs) -> None:
        super(MGDAOptimizer, self).__init__(**kwargs)
        assert type(max_iter) == int, f"{type(max_iter)=}"
        assert max_iter > 0, f"{max_iter=}"
        self.max_iter = max_iter

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
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        alpha = self._frank_wolfe_solver_(grads_list)
        result = sum([alpha_i * g_i for alpha_i, g_i in zip(list(alpha), grads_list)])
        return result

    def _frank_wolfe_solver_(self, grads_list: List[torch.Tensor], threshold: float = 1.0e-03) -> torch.Tensor:
        r"""
        Args:
            threshold (float): serves as the stopping criterion for this sub-problem.
        Returns:
            alpha (torch.Tensor): a 1D tensor containing weights for task gradients.
        """
        # input checks
        assert type(grads_list) == list, f"{type(grads_list)=}"
        assert type(threshold) in [float, int] and threshold >= 0, f"{threshold=}"
        # initialization
        alpha = 1 / self.num_tasks * torch.ones(size=(self.num_tasks,), dtype=torch.float32, device=torch.device('cuda'))
        GTG = utils.gradients.get_gram_matrix(grads_list)
        # main loop
        for _ in range(self.max_iter):
            t_hat = torch.argmin(torch.matmul(GTG, alpha))
            a = grads_list[t_hat]
            b = sum([alpha_i * g_i for alpha_i, g_i in zip(list(alpha), grads_list)])
            gamma_hat = MGDAOptimizer._frank_wolfe_solver_line_(a, b)
            alpha *= (1 - gamma_hat)
            alpha[t_hat] += gamma_hat
            # sanity check
            assert torch.all(torch.maximum(alpha - 1, 0 - alpha) < NUMERICAL_STABILITY), f"{alpha=}"
            alpha = torch.clamp(alpha, min=0, max=1)
            assert abs(alpha.sum() - 1) < NUMERICAL_STABILITY, f"{alpha=}, {alpha.sum()=}"
            alpha = alpha / alpha.sum()
            # early stopping
            if gamma_hat < threshold:
                break
        # return
        return alpha

    @staticmethod
    def _frank_wolfe_solver_line_(a: torch.Tensor, b: torch.Tensor) -> Union[int, torch.Tensor]:
        r"""This function returns the solution to
        :math:`\min_{\gamma \in [0, 1]}\|\gamma a + (1-\gamma) b\|_{2}^{2}`.

        Returns:
            gamma: the scalar that solves the problem.
        """
        # input checks
        assert type(a) == type(b) == torch.Tensor
        assert a.dim() == b.dim() == 1, f"{a.shape=}, {b.shape=}"
        # compute inner products
        aa = torch.dot(a, a)
        bb = torch.dot(b, b)
        ab = torch.dot(a, b)
        # compute minimizer
        if ab >= aa:
            result = 1
        elif ab >= bb:
            result = 0
        else:
            result = ((bb - ab) / (aa + bb - 2*ab))
            # sanity check
            assert max(result - 1, 0 - result) < NUMERICAL_STABILITY, f"{result=}"
            result = torch.clamp(result, min=0, max=1)
        # return
        return result
