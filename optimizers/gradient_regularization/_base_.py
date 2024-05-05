from typing import Tuple, List, Dict, Union, Optional
from abc import abstractmethod, ABC
import torch

from ..mtl_optimizer import MTLOptimizer


class GradientRegularizationBaseOptimizer(MTLOptimizer, ABC):

    def __init__(
        self,
        wrt_rep: Optional[bool] = False,
        per_layer: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(GradientRegularizationBaseOptimizer, self).__init__(**kwargs)
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        self.wrt_rep = wrt_rep
        assert type(per_layer) == bool, f"{type(per_layer)=}"
        self.per_layer = per_layer

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
        # compute grads
        grads_dict: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]] = self._get_grads_all_tasks_(
            losses=losses, shared_rep=shared_rep, wrt_rep=self.wrt_rep, per_layer=self.per_layer,
        )
        # compute loss
        if self.per_layer:
            num_layers = len(list(grads_dict.values())[0])
            assert all(len(grads_dict[name]) == num_layers for name in grads_dict)
            reg_loss: List[torch.Tensor] = [self._gradient_regularization_(
                grads_list=[grads_dict[name][idx] for name in grads_dict],
            ) for idx in range(num_layers)]
            reg_loss: torch.Tensor = torch.stack(reg_loss).mean()
        else:
            reg_loss: torch.Tensor = self._gradient_regularization_(
                grads_list=list(grads_dict.values()),
            )
        tot_loss = sum(list(losses.values())) + reg_loss
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        tot_loss.backward()
