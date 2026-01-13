from dataclasses import asdict, dataclass


@dataclass
class BaseProgressInfo:
    progress_percentage: float

    def to_dict(self) -> dict:
        return asdict(self)
