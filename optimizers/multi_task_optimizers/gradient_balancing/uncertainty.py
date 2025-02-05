from typing import Tuple, Dict, Union
import torch

from ._base_ import GradientBalancingBaseOptimizer


class UncertaintyOptimizer(GradientBalancingBaseOptimizer):
    __doc__ = r"""
    Reference: https://github.com/AvivNavon/AuxiLearn/blob/master/experiments/weight_methods.py#L81
    Retrieved: Sun Aug 20, 2023
    """

    def __init__(self, **kwargs) -> None:
        super(UncertaintyOptimizer, self).__init__(**kwargs)
        self.logvars = None
        self.first_iter = True

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # initialization
        losses_tensor = torch.stack(list(losses.values()))
        if self.first_iter:
            self.logvars = torch.ones_like(losses_tensor)
            self.optimizer.add_param_group({'params': self.logvars})
            assert len(self.optimizer.param_groups) == 2, f"{len(self.optimizer.param_groups)=}"
            self.first_iter = False
        # reweigh losses
        total_loss = (1 / (2 * torch.exp(self.logvars)) * losses_tensor + self.logvars / 2).sum()
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
