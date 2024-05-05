from typing import Tuple, List, Dict, Union, Optional
from abc import abstractmethod, ABC
import torch

from ..mtl_optimizer import MTLOptimizer


class GradientManipulationBaseOptimizer(MTLOptimizer, ABC):

    def __init__(
        self,
        wrt_rep: Optional[bool] = False,
        per_layer: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super(GradientManipulationBaseOptimizer, self).__init__(**kwargs)
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        self.wrt_rep = wrt_rep
        assert type(per_layer) == bool, f"{type(per_layer)=}"
        self.per_layer = per_layer

    @abstractmethod
    def _gradient_manipulation_(
        self,
        grads_list: List[torch.Tensor],
        shared_rep: Optional[torch.Tensor] = None,
    ):
        r"""Each gradient manipulation method will implement its own.
        """
        raise NotImplementedError("_gradient_manipulation_ not implemented for abstract base class.")

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert type(losses) == dict, f"{type(losses)=}"
        assert type(shared_rep) in [torch.Tensor, tuple], f"{type(shared_rep)=}"
        # initialization
        grads_dict: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]] = self._get_grads_all_tasks_(
            losses=losses, shared_rep=shared_rep, wrt_rep=self.wrt_rep, per_layer=self.per_layer,
        )
        if type(shared_rep) == tuple:
            shared_rep = torch.cat([g.flatten() for g in shared_rep])
        else:
            shared_rep = shared_rep.flatten()
        # compute gradients
        with torch.no_grad():
            if self.per_layer:
                num_layers = len(list(grads_dict.values())[0])
                assert all(len(grads_dict[name]) == num_layers for name in grads_dict)
                manipulated_grad: List[torch.Tensor] = [self._gradient_manipulation_(
                    grads_list=[grads_dict[name][idx] for name in grads_dict], shared_rep=shared_rep,
                ) for idx in range(num_layers)]
                manipulated_grad: torch.Tensor = torch.cat(manipulated_grad, dim=0)
            else:
                manipulated_grad: torch.Tensor = self._gradient_manipulation_(
                    grads_list=list(grads_dict.values()), shared_rep=shared_rep,
                )
        assert manipulated_grad.dim() == 1, f"{manipulated_grad.shape=}"
        # populate gradients for task-specific parameters
        self.optimizer.zero_grad(set_to_none=True)
        for p in self._get_shared_params_():
            assert p.requires_grad
            assert p.grad is None
            p.requires_grad = False
        multi_task_loss = list(losses.values())
        avg_loss = sum(multi_task_loss) / len(multi_task_loss)
        avg_loss.backward(retain_graph=self.wrt_rep)
        for p in self._get_shared_params_():
            assert not p.requires_grad
            assert p.grad is None
            p.requires_grad = True
        # populate gradients for shared parameters
        if self.wrt_rep:
            shared_rep.backward(gradient=manipulated_grad)
        else:
            self._set_grad_(grad=manipulated_grad)

    def _set_grad_(self, grad: torch.Tensor) -> None:
        r"""This function sets the gradients of the shared parameters in the model.

        Returns:
            None
        """
        # input checks
        assert type(grad) == torch.Tensor, f"{type(grad)=}"
        assert len(grad.shape) == 1, f"{grad.shape=}"
        # populate gradient
        grad_read_idx = 0
        for p, shape in zip(self._get_shared_params_(), self.shared_params_shapes):
            length = int(torch.prod(torch.tensor(shape)))
            assert p.requires_grad
            assert p.grad is None
            p.grad = grad[grad_read_idx:grad_read_idx+length].view(shape)
            grad_read_idx += length
        assert grad_read_idx == len(grad), f"{grad_read_idx=}, {grad.shape=}"
