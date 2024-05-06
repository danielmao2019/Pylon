from typing import List, Optional
import numpy
import torch
from scipy.optimize import minimize

from ._base_ import GradientManipulationBaseOptimizer
import utils


class CAGradOptimizer(GradientManipulationBaseOptimizer):
    __doc__ = r"""Paper: Conflict-Averse Gradient Descent for Multi-task Learning (https://arxiv.org/abs/2110.14048)

    References
        * https://github.com/Cranial-XIX/CAGrad (Sun Jun 11, 2023)
        * https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/CAGrad.py (Thu Feb 15, 2024)
    """

    def __init__(self, c: Optional[float] = 0.5, **kwargs) -> None:
        super(CAGradOptimizer, self).__init__(**kwargs)
        assert type(c) == float, f"{type(c)=}"
        self.c = c

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
        # initialization
        A = utils.gradients.get_gram_matrix(grads_list).cpu().numpy()
        assert A.shape[:] == (self.num_tasks, self.num_tasks), f"{A.shape=}"
        sqrt_phi: float = self.c * numpy.sqrt(A.mean() + 1e-8).item() + 1e-8
        # solve optimization sub-problem
        x0 = numpy.ones(self.num_tasks) / self.num_tasks
        bounds = tuple((0,1) for _ in range(self.num_tasks))
        constraints=({'type': 'eq', 'fun': lambda x: 1-sum(x)})
        def fun(x):
            wTGTG = x.reshape(1, self.num_tasks).dot(A)
            term1 = wTGTG.dot(numpy.ones((self.num_tasks, 1)) / self.num_tasks)
            term2 = sqrt_phi * numpy.sqrt(wTGTG.dot(x.reshape(self.num_tasks, 1)) + 1e-8)
            obj = (term1 + term2).item()
            return obj
        solution: numpy.ndarray = minimize(fun=fun, x0=x0, bounds=bounds, constraints=constraints).x
        # compute final gradient
        avg_grad = sum(grads_list) / len(grads_list)
        weights_list: List[float] = solution.tolist()
        weighed_grad = sum([weight * grad for (weight, grad) in zip(weights_list, grads_list)])
        lmbda = sqrt_phi / (weighed_grad.norm() + 1e-8)
        final_grad = avg_grad + lmbda * weighed_grad
        assert final_grad.shape == grads_list[0].shape, f"{final_grad.shape=}, {grads_list[0].shape=}"
        return final_grad
