from typing import Dict, Any, Optional
from .runner_detection import detect_runner_type
from .base_progress_tracker import BaseProgressTracker
from .trainer_progress_tracker import TrainerProgressTracker
from .evaluator_progress_tracker import EvaluatorProgressTracker


def create_progress_tracker(work_dir: str, config: Optional[Dict[str, Any]] = None) -> BaseProgressTracker:
    """Factory function to create appropriate progress tracker.
    
    Args:
        work_dir: Path to log/work directory
        config: Optional config dictionary for additional context
        
    Returns:
        Appropriate progress tracker instance (TrainerProgressTracker or EvaluatorProgressTracker)
        
    Raises:
        ValueError: If runner type cannot be determined (delegated to detect_runner_type)
    """
    runner_type = detect_runner_type(work_dir, config)
    
    if runner_type == 'evaluator':
        return EvaluatorProgressTracker(work_dir, config)
    else:
        return TrainerProgressTracker(work_dir, config)