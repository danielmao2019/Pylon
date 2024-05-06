from typing import Tuple, List, Dict, Union, Optional
import torch

from ._base_ import GradientManipulationBaseOptimizer
import utils


class IMTLOptimizer(GradientManipulationBaseOptimizer):

    def __init__(self, **kwargs) -> None:
        super(IMTLOptimizer, self).__init__(**kwargs)
        self.scale_params = torch.zeros(size=(self.num_tasks,), requires_grad=True)

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        r"""Override parent method.
        """
        # initialization
        lr = self.optimizer.param_groups[0]['lr']
        # scale losses
        for idx, task in enumerate(losses.keys()):
            losses[task] = losses[task] * torch.exp(self.scale_params[idx]) - self.scale_params[idx]
        # compute scale parameters update
        update = torch.autograd.grad(
            outputs=sum(list(losses.values())),
            inputs=self.scale_params,
            retain_graph=True,
        )
        assert type(update) == tuple and len(update) == 1
        update = update[0]
        assert type(update) == torch.Tensor and update.shape[:] == (self.num_tasks,)
        # update scale parameters
        self.scale_params = self.scale_params - lr * update
        # do backward as usual
        super().backward(losses=losses, shared_rep=shared_rep)

    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            grads_list: a list of 1D gradient tensors to be manipulated.
        Returns:
            result: the 1D manipulated gradient tensor.
        """
        # input checks
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        # initialization
        random_idx = torch.randint(low=0, high=self.num_tasks, size=(1,)).item()
        device = grads_list[0].device
        # compute
        grad_norms = [torch.linalg.vector_norm(g) for g in grads_list]
        D = [grads_list[random_idx] - grads_list[idx]
             for idx in range(self.num_tasks) if idx != random_idx]
        U = [grads_list[random_idx] / grad_norms[random_idx] - grads_list[idx] / grad_norms[idx]
             for idx in range(self.num_tasks) if idx != random_idx]
        DUT = utils.gradients.get_gram_matrix(grad_list=D, other=U)
        inverse = torch.linalg.inv(DUT)
        alpha = torch.matmul(
            torch.tensor([torch.dot(grads_list[random_idx], u) for u in U]).reshape((1, self.num_tasks-1)).to(device),
            inverse,
        )
        assert alpha.shape == (1, self.num_tasks-1), f"{alpha.shape=}"
        alpha = alpha.reshape((self.num_tasks-1,)).tolist()
        alpha.insert(random_idx, 1-sum(alpha))
        assert len(alpha) == self.num_tasks and sum(alpha) == 1, f"{alpha=}, {self.num_tasks=}, {sum(alpha)=}"
        result = sum([alpha[i] * grads_list[i] for i in range(self.num_tasks)])
        return result
