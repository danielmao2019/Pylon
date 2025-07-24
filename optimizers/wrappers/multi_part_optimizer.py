from typing import Dict, Any, Optional
from optimizers import BaseOptimizer
from utils.builders import build_from_config
from utils.input_checks import check_write_file
from utils.io.json import safe_save_json


class MultiPartOptimizer(BaseOptimizer):

    def __init__(self, optimizer_cfgs: dict) -> None:
        self.optimizers = {
            name: build_from_config(optimizer_cfgs[name])
            for name in optimizer_cfgs
        }
        super(MultiPartOptimizer, self).__init__()

    def reset_buffer(self) -> None:
        for name in self.optimizers:
            self.optimizers[name].reset_buffer()

    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("MultiPartOptimizer.backward is unused and should not be called.")

    # ====================================================================================================
    # ====================================================================================================

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: self.optimizers[name].state_dict()
            for name in self.optimizers
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        for name in self.optimizers:
            self.optimizers[name].load_state_dict(state_dict[name])

    # ====================================================================================================
    # ====================================================================================================

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        r"""Summarize each optimizer.
        """
        result = {
            name: self.optimizers[name].summarize(output_path=None)
            for name in self.optimizers
        }
        if output_path is not None:
            check_write_file(path=output_path)
            safe_save_json(obj=result, filepath=output_path)
        return result
