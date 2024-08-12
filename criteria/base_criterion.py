from typing import List, Dict, Union, Any, Optional
import torch
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer


class BaseCriterion(torch.nn.Module):

    def __init__(self):
        super(BaseCriterion, self).__init__()
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = []

    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""Default __call__ method. This method assumes `_compute_loss_` is implemented and both y_pred
        and y_true are either tensors or dictionaries of one key-val pair.
        """
        assert hasattr(self, '_compute_loss_') and callable(self._compute_loss_)
        # input checks
        if type(y_pred) == dict:
            assert len(y_pred) == 1, f"{y_pred.keys()=}"
            y_pred = list(y_pred.values())[0]
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        assert type(y_true) == torch.Tensor, f"{type(y_true)=}"
        # compute loss
        loss = self._compute_loss_(y_pred=y_pred, y_true=y_true)
        assert type(loss) == torch.Tensor, f"{type(loss)=}"
        assert loss.ndim == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss

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
