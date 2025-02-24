from typing import List, Optional
import numpy
import torch

from ._base_ import GradientManipulationBaseOptimizer
import utils


NUMERICAL_STABILITY = 1.0e-05


class NashMTLOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Multi-Task Learning as a Bargaining Game (https://arxiv.org/abs/2202.01017)
    """

    def __init__(self, max_iter: Optional[int] = 20, **kwargs) -> None:
        super(NashMTLOptimizer, self).__init__(**kwargs)
        assert type(max_iter) == int, f"{type(max_iter)=}"
        assert max_iter > 0, f"{max_iter=}"
        self.max_iter = max_iter
        self._init_problem_()

    def _init_problem_(self):
        # import here to avoid dependency
        import cvxpy as cp
        # define variables
        self.alpha_var = cp.Variable(shape=(self.num_tasks,), nonneg=True)
        self.alpha = cp.Parameter(shape=(self.num_tasks,), value=numpy.ones(self.num_tasks, dtype=numpy.float32))
        self.gtg = cp.Parameter(shape=(self.num_tasks, self.num_tasks), value=numpy.eye(self.num_tasks))
        # define constraints
        beta_var = self.gtg @ self.alpha_var
        phi_var = cp.log(self.alpha_var) + cp.log(beta_var)
        constraints = []
        for i in range(self.num_tasks):
            constraints.append(self.alpha_var[i] >= NUMERICAL_STABILITY)
            constraints.append(beta_var >= NUMERICAL_STABILITY)
            constraints.append(phi_var[i] >= 0)
        # define objective
        beta = self.gtg @ self.alpha
        phi = cp.sum(cp.log(self.alpha) + cp.log(beta))
        gradient = 1 / self.alpha + (1 / beta) @ self.gtg
        phi_linear = phi + gradient @ (self.alpha_var - self.alpha)
        obj = cp.Minimize(cp.sum(beta_var) + phi_linear)
        # define problem
        self.problem = cp.Problem(obj, constraints)

    def _stopping_criterion_(self):
        return (
            (self.alpha_var.value is None)
            or (numpy.linalg.norm(self.gtg.value @ self.alpha_var.value - 1 / (self.alpha_var.value + 1e-10)) < 1e-3)
            or (numpy.linalg.norm(self.alpha_var.value - self.alpha.value) < 1e-6)
        )

    def solve_optimization(self):
        # propose a new solution
        for _ in range(self.max_iter):
            try:
                self.problem.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_var.value = self.alpha.value
            if self._stopping_criterion_():
                break
        # accept proposal
        assert self.alpha_var.value is not None
        assert numpy.all(self.alpha_var.value >= 0)
        self.alpha.value = self.alpha_var.value

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
        # compute weights
        gtg = utils.gradients.get_gram_matrix(grads_list)
        self.gtg.value = gtg.detach().cpu().numpy()
        self.solve_optimization()
        # normalize
        self.alpha.value /= self.alpha.value.sum()
        # re-weigh gradients
        result = sum([grads_list[i] * self.alpha.value[i] for i in range(self.num_tasks)])
        return result
