from typing import Tuple, List, Dict, Union, Optional
import time
import torch
from optimizers import BaseOptimizer
from utils.builders import build_from_config
from utils.logging.text_logger import TextLogger


class MTLOptimizer(BaseOptimizer):
    __doc__ = r"""A hook contains custom operations for the optimizer.
    """

    def __init__(
        self,
        optimizer_config: dict,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        logger: Optional[TextLogger] = None,
        **kwargs,
    ):
        r"""
        Args:
            optimizer_config (dict): a config dict for building the core optimizer.
            losses (Dict[str, torch.Tensor]): a dummy loss dictionary for initialization purpose.
            shared_rep (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): a dummy shared representation for initialization purpose.
            logger (utils.logging.logger.TextLogger).
            kwargs (dict): other unused arguments. e.g., wrt_rep and per_layer for gradient balancing methods.
        """
        super(MTLOptimizer, self).__init__()
        self.optimizer = build_from_config(config=optimizer_config)
        self.logger = logger if logger is not None else TextLogger()
        self._init_shared_params_mask_(loss_dict=losses, shared_rep=shared_rep)
        self._init_shared_params_shapes_()
        self.num_tasks: int = len(losses)

    # ====================================================================================================
    # shared parameters related methods
    # ====================================================================================================

    def _init_shared_params_mask_(
        self,
        loss_dict: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        r"""This method initializes `self.shared_params_mask`, a 1D boolean tensor where 1 means the
        current parameters are shared. Order is defined by the double for loop
        "for group in self.optimizer.param_groups for p in group['params']".
        """
        # input checks
        assert type(loss_dict) == dict, f"{type(loss_dict)=}"
        assert all(type(key) == str for key in loss_dict.keys())
        assert all([
            type(value) == torch.Tensor and value.ndim == 0 and value.requires_grad
            for value in loss_dict.values()
        ])
        if type(shared_rep) == torch.Tensor:
            shared_rep = (shared_rep,)
        assert type(shared_rep) == tuple
        assert all(type(elem) == torch.Tensor for elem in shared_rep)
        shared_rep = torch.cat([g.flatten() for g in shared_rep])
        assert shared_rep.ndim == 1, f"{shared_rep.shape=}"
        # compute gradients with method 1
        self.optimizer.zero_grad(set_to_none=True)
        dummy_gradient = torch.zeros_like(shared_rep)
        shared_rep.backward(gradient=dummy_gradient, retain_graph=True)
        shared_params_mask_v1 = torch.tensor([
            p.requires_grad and p.grad is not None
            for group in self.optimizer.param_groups for p in group['params']
        ], dtype=torch.bool, device=torch.device('cuda'))
        # compute gradients with method 2
        masks: List[torch.Tensor] = []
        for loss in loss_dict.values():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            masks.append(torch.tensor([
                p.requires_grad and p.grad is not None
                for group in self.optimizer.param_groups for p in group['params']
            ], dtype=torch.bool, device=torch.device('cuda')))
        shared_params_mask_v2 = torch.prod(torch.stack(masks).type(torch.int64), dim=0).type(torch.bool)
        # sanity check
        assert torch.equal(shared_params_mask_v1, shared_params_mask_v2)
        # assign to class attribute
        self.shared_params_mask: torch.Tensor = shared_params_mask_v1

    def _get_shared_params_(self):
        r"""Generator function for the shared parameters.
        """
        idx = 0
        for group in self.optimizer.param_groups:
            if len(group['params']) == 1:
                continue
            for p in group['params']:
                if self.shared_params_mask[idx]:
                    yield p
                idx += 1
        assert idx == len(self.shared_params_mask), f"{idx=}, {len(self.shared_params_mask)=}"

    def _init_shared_params_shapes_(self) -> None:
        r"""This method initializes `self.shared_params_shapes`, a list of `torch.Size` for the shapes
        of the shared parameters.
        """
        shapes: List[torch.Size] = [p.shape for p in self._get_shared_params_()]
        # sanity check on mask and shapes
        assert hasattr(self, 'shared_params_mask') and type(self.shared_params_mask) == torch.Tensor
        shared_count = self.shared_params_mask.type(torch.int64).sum()
        assert shared_count == len(shapes), f"{shared_count=}, {len(shapes)=}"
        # assign to class attribute
        self.shared_params_shapes: List[torch.Size] = shapes

    # ====================================================================================================
    # gradient computation methods
    # ====================================================================================================

    def _get_grad_par_(self, loss: torch.Tensor, per_layer: bool) -> Union[torch.Tensor, List[torch.Tensor]]:
        r"""Get gradient of the given (single) loss w.r.t. shared parameters.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
            per_layer (bool): if True, gradient w.r.t. each parameter group will not be concatenated.
        Returns:
            grad (torch.Tensor or List[torch.Tensor]): the gradient of loss w.r.t. shared parameters.
        """
        # compute gradients
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad: List[torch.Tensor] = [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in self._get_shared_params_()
        ]
        assert len(grad) == len(self.shared_params_shapes)
        grad = [g.flatten() for g in grad]
        if not per_layer:
            grad = torch.cat(grad, dim=0)
            assert grad.ndim == 1, f"{grad.shape=}"
        return grad

    @staticmethod
    def _get_grad_rep_(
        loss: torch.Tensor,
        shared_rep: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        r"""Get gradient of the given (single) loss w.r.t. shared representation.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
        Returns:
            grad (torch.Tensor): the gradient of loss w.r.t. shared representation.
        """
        # initialization
        if type(shared_rep) == torch.Tensor:
            shared_rep = (shared_rep,)
        # compute gradients
        grad_seq: Tuple[torch.Tensor, ...] = torch.autograd.grad(
            outputs=[loss], inputs=shared_rep, allow_unused=True, retain_graph=True,
        )
        assert type(grad_seq) == tuple, f"{type(grad_seq)=}"
        assert len(grad_seq) == len(shared_rep), f"{len(grad_seq)=}, {len(shared_rep)=}"
        grad_seq: List[torch.Tensor] = list(grad_seq)
        for idx in range(len(grad_seq)):
            if grad_seq[idx] is None:
                grad_seq[idx] = torch.zeros_like(shared_rep[idx])
            assert type(grad_seq[idx]) == torch.Tensor, f"{idx=}, {type(grad_seq[idx])=}"
            assert grad_seq[idx].shape == shared_rep[idx].shape, f"{grad_seq[idx].shape=}, {shared_rep[idx].shape=}"
        grad: torch.Tensor = torch.cat([g.flatten() for g in grad_seq], dim=0)
        assert grad.ndim == 1, f"{grad.shape=}"
        return grad

    def _get_grads_all_tasks_(
        self,
        loss_dict: Dict[str, torch.Tensor],
        shared_rep: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        wrt_rep: Optional[bool] = False,
        per_layer: Optional[bool] = False,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        r"""This method computes the gradients for all losses.
        """
        # input checks
        assert type(loss_dict) == dict, f"{type(loss_dict)=}"
        assert all(type(key) == str for key in loss_dict.keys())
        assert all([
            type(value) == torch.Tensor and value.ndim == 0 and value.requires_grad
            for value in loss_dict.values()
        ])
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        assert type(per_layer) == bool, f"{type(per_layer)=}"
        if wrt_rep:
            assert shared_rep is not None
            if type(shared_rep) != torch.Tensor:
                assert type(shared_rep) == tuple, f"{type(shared_rep)=}"
                assert all(type(elem) == torch.Tensor for elem in shared_rep)
        # initialize time
        grad_time = time.time()
        if wrt_rep:
            grads_dict = {
                name: self._get_grad_rep_(loss=loss_dict[name], shared_rep=shared_rep)
                for name in loss_dict
            }
        else:
            grads_dict = {
                name: self._get_grad_par_(loss=loss_dict[name], per_layer=per_layer)
                for name in loss_dict
            }
        self.logger.update_buffer({'grad_time': time.time() - grad_time})
        return grads_dict

    # ====================================================================================================
    # ====================================================================================================

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        r"""The default backward method.

        Args:
            shared_rep (Union[torch.Tensor, Tuple]): unused argument.
        """
        self.optimizer.zero_grad(set_to_none=True)
        losses_tensor = torch.stack(list(losses.values()))
        avg_loss = losses_tensor.mean()
        avg_loss.backward()
