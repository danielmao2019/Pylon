from typing import List, Literal
import os
from .base_progress_tracker import BaseProgressTracker, ProgressInfo


class EvaluatorProgressTracker(BaseProgressTracker):
    """Progress tracker for BaseEvaluator runs."""
    
    def get_runner_type(self) -> Literal['evaluator']:
        return 'evaluator'
    
    def get_expected_files(self) -> List[str]:
        return ["evaluation_scores.json"]
    
    def get_log_pattern(self) -> str:
        return "eval_*.log"
    
    def calculate_progress(self) -> ProgressInfo:
        """Calculate evaluator-specific progress."""
        # Check if evaluation is complete
        eval_complete = self.is_complete()
        
        # Binary progress: 0% or 100%
        progress_percentage = 100.0 if eval_complete else 0.0
        
        # For evaluators: completed_epochs = 1 if finished, 0 if not
        completed_epochs = 1 if eval_complete else 0
        
        return ProgressInfo(
            # Existing fields (adapted for evaluator)
            completed_epochs=completed_epochs,
            progress_percentage=progress_percentage,
            early_stopped=False,  # Not applicable for evaluators
            early_stopped_at_epoch=None,
            
            # New fields
            runner_type='evaluator',
            total_epochs=1,  # Evaluators have conceptually 1 "epoch"
        )
    
    def is_complete(self) -> bool:
        """Check if evaluator is complete."""
        return all(
            os.path.exists(os.path.join(self.work_dir, f)) and
            os.path.getsize(os.path.join(self.work_dir, f)) > 0
            for f in self.get_expected_files()
        )