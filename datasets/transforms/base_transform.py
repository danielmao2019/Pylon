from typing import List, Dict, Union, Any
from abc import ABC, abstractmethod
import torch


class BaseTransform(ABC):

    def __call__(self, *args) -> Union[
        Dict[str, Dict[str, Any]], torch.Tensor, List[torch.Tensor],
    ]:
        if len(args) == 1:
            return self._call_concrete_(args[0])
        else:
            assert all(type(arg) == torch.Tensor for arg in args)
            return [self._call_concrete_(arg) for arg in args]

    @abstractmethod
    def _call_concrete_(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("_call_concrete_ not implemented for abstract base class.")
