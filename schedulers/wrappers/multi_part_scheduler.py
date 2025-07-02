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

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: self.schedulers[name].state_dict()
            for name in self.schedulers
        }

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        for name in self.schedulers:
            self.schedulers[name].load_state_dict(state_dict[name])
