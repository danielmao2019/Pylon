from typing import Dict, Union, Optional
import torch
from criteria import BaseCriterion
from utils.input_checks import check_write_file


class SingleTaskCriterion(BaseCriterion):

    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""This method assumes `_compute_loss` is implemented and both y_pred
        and y_true are either tensors or dictionaries of one key-val pair.
        """
        assert hasattr(self, '_compute_loss') and callable(self._compute_loss)
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
        loss = self._compute_loss(y_pred=y_pred, y_true=y_true)
        self.add_to_buffer(loss)
        return loss

    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        r"""This method stacks loss trajectory across all data points in buffer.
        """
        if not self.use_buffer or self.buffer is None:
            raise ValueError("Buffer is disabled for this criterion")
        assert len(self.buffer) != 0
        # summarize losses
        result = torch.stack(self.buffer, dim=0)
        assert result.ndim == 1, f"{result.shape=}"
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            torch.save(obj=result, f=output_path)
        return result
