from typing import Tuple, List, Dict, Union, Optional
import torch

from ._base_ import GradientBalancingBaseOptimizer


class GradNormOptimizer(GradientBalancingBaseOptimizer):
    __doc__ = r"""
    Reference: https://github.com/AvivNavon/AuxiLearn/blob/master/experiments/weight_methods.py#L81
    Retrieved: Sun Aug 20, 2023
    """

    def __init__(self, alpha: Optional[float] = 1.5, **kwargs) -> None:
        super(GradNormOptimizer, self).__init__(**kwargs)
        assert type(alpha) == float, f"{type(alpha)=}"
        self.alpha = alpha
        self.weights = torch.ones(size=(self.num_tasks,), requires_grad=True, dtype=torch.float32, device=torch.device('cuda'))
        self.weights_optimizer = torch.optim.SGD(
            params=[self.weights], lr=1.0e-05, momentum=0.9, weight_decay=0,
        )
        self.first_iter = True

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert len(losses) == self.num_tasks, f"{len(losses)=}, {self.num_tasks=}"

        # initialization
        grads_all_tasks: Dict[str, torch.Tensor] = self._get_grads_all_tasks_(losses=losses, shared_rep=shared_rep, wrt_rep=False)
        grads_all_tasks: List[torch.Tensor] = list(grads_all_tasks.values())
        losses_tensor = torch.stack(list(losses.values()))
        if self.first_iter:
            self.init_losses = losses_tensor.detach().data
            self.first_iter = False

        # compute and populate gradients for task weights
        grad_norms = torch.stack([
            torch.norm(w_i * g_i) for w_i, g_i in zip(list(self.weights), grads_all_tasks)
        ])
        with torch.no_grad():
            loss_ratios = losses_tensor / self.init_losses
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            grad_norms_expected = grad_norms.mean() * (inverse_train_rates ** self.alpha)
        grad_norm_loss = (grad_norms - grad_norms_expected).abs().sum()
        self.weights_optimizer.zero_grad(set_to_none=True)
        assert self.weights.requires_grad == True
        assert self.weights.grad is None, f"{self.weights.grad=}"
        grad_norm_loss.backward(retain_graph=True)
        assert self.weights.requires_grad == True
        assert self.weights.grad is not None
        self.weights_optimizer.step()
        with torch.no_grad():
            self.weights = torch.nn.functional.softmax(self.weights, dim=0)
        assert self.weights.requires_grad == False
        assert self.weights.grad is None

        # compute and populate gradients for network parameters
        assert torch.all(self.weights >= 0) and (self.weights.sum() - 1).abs() < 1e-05, f"{self.weights=}, {self.weights.sum()=}"
        weighted_sum_loss = (self.weights * losses_tensor).sum()
        self.optimizer.zero_grad()
        weighted_sum_loss.backward()
        assert self.weights.requires_grad == False
        self.weights.requires_grad_()
