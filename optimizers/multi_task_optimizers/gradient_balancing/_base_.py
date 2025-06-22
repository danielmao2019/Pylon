from typing import Tuple, Dict, Union
from abc import ABC, abstractmethod
import torch
from optimizers.multi_task_optimizers.mtl_optimizer import MTLOptimizer


class GradientBalancingBaseOptimizer(MTLOptimizer, ABC):

    @abstractmethod
    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        r"""
        Each gradient balancing method will implement its own.
        """
        raise NotImplementedError()
