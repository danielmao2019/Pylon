from typing import List, Dict, Union, Any, Optional
from abc import ABC, abstractmethod
import torch
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer


class BaseCriterion(ABC, torch.nn.Module):

    def __init__(self):
        super(BaseCriterion, self).__init__()
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = []

    @abstractmethod
    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Returns:
            loss (torch.Tensor or Dict[str, torch.Tensor]): a scalar tensor for single loss
                or dictionary of scalar tensors for multiple losses.
        """
        raise NotImplementedError("[ERROR] __call__ not implemented for abstract base class.")

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        r"""Default summary: trajectory of losses across all examples in buffer.
        """
        if output_path is not None:
            check_write_file(path=output_path)
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            if type(self.buffer[0]) == torch.Tensor:
                trajectory = torch.stack(self.buffer)
                assert trajectory.dim() == 1, f"{trajectory.shape=}"
                result['loss_trajectory'] = trajectory
            elif type(self.buffer[0]) == dict:
                buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
                for key in buffer:
                    key_losses = torch.stack(buffer[key], dim=0)
                    assert key_losses.dim() == 1, f"{key_losses.shape=}"
                    result[f"loss_{key}_trajectory"] = key_losses
            else:
                raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
        if output_path is not None:
            torch.save(obj=result, f=output_path)
        return result
