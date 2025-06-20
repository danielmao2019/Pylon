from typing import Dict, Any
from utils.builders import build_from_config


class MultiPartScheduler:

    def __init__(self, scheduler_cfgs: Dict[str, Any]) -> None:
        self.schedulers = {
            name: build_from_config(scheduler_cfgs[name])
            for name in scheduler_cfgs
        }

    # ====================================================================================================
    # ====================================================================================================

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            name: self.schedulers[name].state_dict(*args, **kwargs)
            for name in self.schedulers
        }

    def load_state_dict(self, *args: Any, **kwargs: Any) -> None:
        for name in self.schedulers:
            self.schedulers[name].load_state_dict(*args, **kwargs)
