from typing import Tuple, Dict, Union
import torch

from ._base_ import GradientBalancingBaseOptimizer


class GLSOptimizer(GradientBalancingBaseOptimizer):
    __doc__ = r"""Paper: MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (https://arxiv.org/abs/1904.08492)
    """

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert len(losses) == self.num_tasks, f"{len(losses)=}, {self.num_tasks=}"
        # initialization
        losses_tensor = torch.stack(list(losses.values()))
        # compute geometric loss
        geo_loss = torch.pow(losses_tensor, 1/self.num_tasks).prod()
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        geo_loss.backward()
