from typing import Tuple, List, Dict, Union
import torch
import time
import logging
from utils.builder import build_from_config


class MTLOptimizer:
    __doc__ = r"""A hook contains custom operations for the optimizer.
    """

    def __init__(
        self,
        optimizer_config: dict,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
        logger: logging.Logger,
        **kwargs,
    ):
        r"""
        Args:
            losses (Dict[str, torch.Tensor]): a dummy loss dictionary for initialization purpose.
            shared_rep (Union[torch.Tensor, Tuple]): a dummy shared representation for initialization purpose.
            kwargs (dict): other unused arguments. e.g., wrt_rep and per_layer for gradient balancing methods.
        """
        self.optimizer = build_from_config(config=optimizer_config)
        self.logger = logger
        self.shared_params_mask = None
        self.shared_params_shapes = None
        torch.autograd.set_detect_anomaly(True)
        self._init_shared_params_mask_(losses=losses, shared_rep=shared_rep)
        self._init_shared_params_shapes_()
        self.num_tasks: int = len(losses)

    # ====================================================================================================
    # initialization methods
    # ====================================================================================================

    def _init_shared_params_mask_(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> torch.Tensor:
        r"""
        self.shared_mask (torch.Tensor): a 1D boolean tensor where 1 means the current parameters are shared.
            order is defined by the double for loop
            "for group in self.optimizer.param_groups for p in group['params']".
        """
        # compute gradients with method 1
        self.optimizer.zero_grad(set_to_none=True)
        shared_rep = torch.cat([g.flatten() for g in shared_rep])
        dummy_gradient = torch.zeros_like(shared_rep)
        shared_rep.backward(gradient=dummy_gradient, retain_graph=True)
        shared_params_mask_v1 = torch.tensor([
            p.requires_grad and p.grad is not None
            for group in self.optimizer.param_groups for p in group['params']
        ], dtype=torch.bool, device=torch.device('cuda'))
        # compute gradients with method 2
        masks = []
        for loss in losses.values():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            masks.append(torch.tensor([
                p.requires_grad and p.grad is not None
                for group in self.optimizer.param_groups for p in group['params']
            ], dtype=torch.int8, device=torch.device('cuda')))
        shared_params_mask_v2 = torch.prod(torch.stack(masks), dim=0).type(torch.bool)
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
        r"""
        Returns:
            shapes: the list of shapes of shared parameters.
        """
        shapes = [p.shape for p in self._get_shared_params_()]
        # sanity check on mask and shapes
        assert hasattr(self, 'shared_params_mask') and type(self.shared_params_mask) == torch.Tensor
        shared_count = self.shared_params_mask.type(torch.int64).sum().item()
        assert shared_count == len(shapes), f"{shared_count=}, {len(shapes)=}"
        # assign to class attribute
        self.shared_params_shapes: List[torch.Size] = shapes

    # ====================================================================================================
    # gradient computation methods
    # ====================================================================================================

    def _get_grad_params_(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Get gradient of the given (single) loss w.r.t. shared parameters.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
        Returns:
            grad (torch.Tensor): the gradient of loss w.r.t. shared parameters.
        """
        # input checks
        assert type(loss) == torch.Tensor, f"{type(loss)=}"
        assert len(loss.shape) == 0, f"{loss.shape=}"
        assert loss.requires_grad
        # compute gradients
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad: List[torch.Tensor] = [
            p.grad if p.grad is not None else torch.zeros_like(p)
            for p in self._get_shared_params_()
        ]
        assert len(grad) == len(self.shared_params_shapes)
        grad = torch.cat([g.flatten() for g in grad])
        assert len(grad.shape) == 1, f"{grad.shape=}"
        return grad

    @staticmethod
    def _get_grad_rep_(
        loss: torch.Tensor,
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> torch.Tensor:
        r"""Get gradient of the given (single) loss w.r.t. shared representation.

        Args:
            loss (torch.Tensor): the value of the (single) loss function at the current iteration.
        Returns:
            grad (torch.Tensor): the gradient of loss w.r.t. shared representation.
        """
        # input checks
        assert type(loss) == torch.Tensor, f"{type(loss)=}"
        assert len(loss.shape) == 0, f"{loss.shape=}"
        assert loss.requires_grad
        if type(shared_rep) == torch.Tensor:
            shared_rep = (shared_rep,)
        assert type(shared_rep) == tuple, f"{type(shared_rep)=}"
        # compute gradients
        grad: List[torch.Tensor] = list(torch.autograd.grad(
            outputs=[loss], inputs=shared_rep, allow_unused=True, retain_graph=True,
        ))
        for idx in range(len(grad)):
            if grad[idx] is None:
                grad[idx] = torch.zeros_like(shared_rep[idx])
            assert type(grad[idx]) == torch.Tensor, f"{idx=}, {type(grad[idx])=}"
            assert grad[idx].shape == shared_rep[idx].shape, f"{grad[idx].shape=}, {shared_rep[idx].shape=}"
        assert len(grad) == len(shared_rep), f"{len(grad)=}, {len(shared_rep)=}"
        grad = torch.cat([g.flatten() for g in grad])
        assert len(grad.shape) == 1, f"{grad.shape=}"
        return grad

    def _get_grads_all_tasks_(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
        wrt_rep: bool,
    ) -> List[torch.Tensor]:
        r"""This function returns the gradients of each task loss in a list.
        """
        # input checks
        assert type(losses) == dict, f"{type(losses)=}"
        assert type(shared_rep) in [torch.Tensor, tuple], f"{type(shared_rep)=}"
        assert type(wrt_rep) == bool, f"{type(wrt_rep)=}"
        # initialize time
        grad_time = time.time()
        method = self._get_grad_rep_ if wrt_rep else self._get_grad_params_
        grads_list = [method(loss=loss, shared_rep=shared_rep) for loss in losses.values()]
        self.logger.update_buffer({'grad_time': time.time() - grad_time})
        return grads_list

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

    # ====================================================================================================
    # ====================================================================================================

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optimizer.step(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.optimizer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.optimizer.load_state_dict(*args, **kwargs)
