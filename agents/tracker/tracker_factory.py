from typing import Dict, Any, Optional
from .runner_detection import detect_runner_type
from .base_tracker import BaseTracker
from .trainer_tracker import TrainerTracker
from .evaluator_tracker import EvaluatorTracker


def create_tracker(work_dir: str, config: Optional[Dict[str, Any]] = None) -> BaseTracker:
    """Factory function to create appropriate progress tracker.
    
    Args:
        work_dir: Path to log/work directory
        config: Optional config dictionary for additional context
        
    Returns:
        Appropriate tracker instance (TrainerTracker or EvaluatorTracker)
        
    Raises:
        ValueError: If runner type cannot be determined (delegated to detect_runner_type)
    """
    runner_type = detect_runner_type(work_dir, config)
    
    if runner_type == 'evaluator':
        return EvaluatorTracker(work_dir, config)
    else:
        return TrainerTracker(work_dir, config)
