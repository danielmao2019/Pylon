from typing import Optional, Literal
from dataclasses import dataclass, asdict


@dataclass
class ProgressInfo:
    """Simplified progress information focusing on progress tracking only.

    Copied from `agents.tracker.base_tracker.ProgressInfo` for the manager module.
    """
    completed_epochs: int
    progress_percentage: float
    early_stopped: bool = False
    early_stopped_at_epoch: Optional[int] = None
    runner_type: Literal['trainer', 'evaluator', 'multi_stage'] = 'trainer'
    total_epochs: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

