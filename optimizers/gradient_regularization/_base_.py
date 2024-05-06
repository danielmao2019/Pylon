from typing import Tuple, List, Dict, Union, Optional
from abc import abstractmethod, ABC
import torch

from ..mtl_optimizer import MTLOptimizer


class GradientRegularizationBaseOptimizer(MTLOptimizer, ABC):

    def __init__(
        self,
        wrt_rep: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(GradientRegularizationBaseOptimizer, self).__init__(**kwargs)
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        self.wrt_rep = wrt_rep

    @abstractmethod
    def _gradient_regularization_(
        self,
        grads_list: List[torch.Tensor],
    ) -> torch.Tensor:
        r"""Each gradient regularization method will implement its own regularization loss.
        """
        raise NotImplementedError("_gradient_regularization_ not implemented for abstract base class.")

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # initialization
        grads_list = self._get_grads_all_tasks_(losses=losses, shared_rep=shared_rep, wrt_rep=self.wrt_rep)
        # sanity check
        assert type(grads_list) == list
        assert len(grads_list) == self.num_tasks, f"{len(grads_list)=}, {self.num_tasks=}"
        for grad in grads_list:
            assert type(grad) == torch.Tensor, f"{type(grad)=}"
        # compute loss
        reg_loss = self._gradient_regularization_(grads_list)
        tot_loss = sum(list(losses.values())) + reg_loss
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        tot_loss.backward()
