from typing import Tuple, List , Dict, Union, Optional
import torch

from ._base_ import GradientBalancingBaseOptimizer


class DWAOptimizer(GradientBalancingBaseOptimizer):

    def __init__(self, window_size: Optional[int] = 32, **kwargs) -> None:
        super(DWAOptimizer, self).__init__(**kwargs)
        assert type(window_size) == int, f"{type(window_size)=}"
        assert window_size > 0, f"{window_size=}"
        self.window_size = window_size
        self.losses_buffer: List[torch.Tensor] = []

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        shared_rep: Union[torch.Tensor, Tuple],
    ) -> None:
        # input checks
        assert len(self.losses_buffer) <= 2 * self.window_size
        losses_tensor = torch.stack(list(losses.values()))
        with torch.no_grad():
            if type(self.losses_buffer) == torch.Tensor:
                assert self.losses_buffer.shape == (2 * self.window_size, self.num_tasks)
                # compute weights
                past = self.losses_buffer[self.window_size:, :].mean(dim=0)
                past_past = self.losses_buffer[:self.window_size, :].mean(dim=0)
                weights = past / (past_past + 1.0e-09)
                # update buffer
                self.losses_buffer[:-1, :] = self.losses_buffer.clone()[1:, :]
                self.losses_buffer[-1, :] = losses_tensor.detach().clone()
            elif type(self.losses_buffer) == list:
                # if buffer not full, then use uniform weighting
                weights = torch.zeros(size=(self.num_tasks,), dtype=torch.float32, device=torch.device('cuda'))
                self.losses_buffer.append(losses_tensor.detach().clone())
                if len(self.losses_buffer) == 2 * self.window_size:
                    self.losses_buffer = torch.stack(self.losses_buffer, dim=0)
            else:
                assert 0, "Should not reach this line."
            # normalize the weights into a probability distribution
            weights = torch.nn.Softmax(dim=0)(weights)
        assert weights.shape == (self.num_tasks,) and not weights.requires_grad
        # reweigh losses
        total_loss = (weights * losses_tensor).sum()
        # populate gradients
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
