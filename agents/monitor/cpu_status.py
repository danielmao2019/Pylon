from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from agents.monitor.process_info import ProcessInfo


@dataclass
class CPUStatus:
    """Status information for a CPU/server."""
    server: str
    window_size: Optional[int]
    max_memory: Optional[int] = None  # Total system memory in MB
    cpu_cores: Optional[int] = None  # Number of CPU cores
    processes: List[ProcessInfo] = field(default_factory=list)
    memory_window: List[int] = field(default_factory=list)  # Memory usage in MB
    cpu_window: List[Optional[float]] = field(default_factory=list)  # CPU utilization percentage
    load_window: List[float] = field(default_factory=list)  # Load average (1min)
    memory_stats: Optional[Dict[str, Optional[float]]] = None
    cpu_stats: Optional[Dict[str, Optional[float]]] = None
    load_stats: Optional[Dict[str, Optional[float]]] = None
    connected: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
