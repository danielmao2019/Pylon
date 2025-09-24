from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from agents.monitor.process_info import ProcessInfo


@dataclass
class GPUStatus:
    """Status information for a GPU."""
    server: str
    index: int
    window_size: Optional[int]
    max_memory: Optional[int] = None
    processes: List[ProcessInfo] = field(default_factory=list)
    memory_window: List[int] = field(default_factory=list)
    util_window: List[int] = field(default_factory=list)
    memory_stats: Optional[Dict[str, Optional[float]]] = None
    util_stats: Optional[Dict[str, Optional[float]]] = None
    connected: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
