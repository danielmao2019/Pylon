from typing import List, Dict, Union
from abc import ABC, abstractmethod
import os
import torch


class BaseCriterion(ABC):

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[torch.Tensor] = []

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Returns:
            loss (torch.Tensor or Dict[str, torch.Tensor]): a scalar tensor for single loss
                or dictionary of scalar tensors for multiple losses.
        """
        raise NotImplementedError("[ERROR] __call__ not implemented for abstract base class.")

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""Default summary: trajectory of losses across all examples in buffer.
        """
        summary: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            if type(self.buffer[0]) == torch.Tensor:
                summary['loss_trajectory'] = torch.cat(self.buffer)
                assert len(summary['loss_trajectory'].shape) == 1, f"{summary.shape=}"
            elif type(self.buffer[0]) == dict:
                keys = self.buffer[0].keys()
                for losses in self.buffer:
                    assert losses.keys() == keys
                    for key in keys:
                        summary[f"loss_{key}_trajectory"] = summary.get(f"loss_{key}_trajectory", []) + [losses[key]]
                for key in keys:
                    summary[f"loss_{key}_trajectory"] = torch.cat(summary[f"loss_{key}_trajectory"])
                    assert len(summary[f"loss_{key}_trajectory"].shape) == 1
            else:
                raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
        if output_path is not None and os.path.isfile(output_path):
            torch.save(obj=summary, f=output_path)
        return summary
