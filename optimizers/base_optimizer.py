from typing import List, Any, Optional, Dict
from abc import ABC, abstractmethod
import torch
from utils.input_checks import check_write_file
from utils.io.json import save_json


class BaseOptimizer(ABC):

    optimizer: torch.optim.Optimizer

    def __init__(self) -> None:
        self.reset_buffer()

    def reset_buffer(self) -> None:
        self.buffer: List[Any] = []

    @abstractmethod
    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Abstract method BaseOptimizer.backward not implemented.")

    # ====================================================================================================
    # ====================================================================================================

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.step(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.optimizer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self.optimizer.load_state_dict(*args, **kwargs)

    # ====================================================================================================
    # ====================================================================================================

    def summarize(self, output_path: Optional[str] = None) -> Any:
        r"""Default summarize method, assuming nothing has been logged to buffer.
        """
        assert len(self.buffer) == 0
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=self.buffer, filepath=output_path)
        return self.buffer
