from typing import Dict, Any, Optional
from optimizers import BaseOptimizer
from utils.builder import build_from_config
from utils.input_checks import check_write_file
from utils.io import save_json


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

    # ====================================================================================================
    # ====================================================================================================

    def state_dict(self, *args, **kwargs) -> dict:
        return {
            name: self.optimizers[name].state_dict(*args, **kwargs)
            for name in self.optimizers
        }

    def load_state_dict(self, *args, **kwargs) -> None:
        for name in self.optimizers:
            self.optimizers[name].load_state_dict(*args, **kwargs)

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
            save_json(obj=result, filepath=output_path)
        return result
