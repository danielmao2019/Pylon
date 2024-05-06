from typing import List, Optional
import torch

from ._base_ import GradientRegularizationBaseOptimizer
import utils


class CosRegOptimizer(GradientRegularizationBaseOptimizer):
    __doc__ = r"""Paper: Regularizing Deep Multi-Task Networks using Orthogonal Gradients (https://arxiv.org/abs/1912.06844)
    """

    def __init__(self, penalty: Optional[float] = 10.0, **kwargs) -> None:
        super(CosRegOptimizer, self).__init__(**kwargs)
        assert type(penalty) == float, f"{type(penalty)=}"
        self.penalty = penalty

    def _gradient_regularization_(self, grad_list: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            grad_list: the list of 1D gradient tensors of each objective.
        Returns:
            reg_loss: the regularization term in the loss function.
        """
        assert len(grad_list) == self.num_tasks, f"{len(grad_list)=}, {self.num_tasks=}"
        cosine_matrix = utils.gradients.get_cosine_matrix(grad_list)
        reg_loss = ((cosine_matrix ** 2).sum() - self.num_tasks) / (self.num_tasks ** 2 - self.num_tasks)
        return self.penalty * reg_loss
