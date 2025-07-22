from typing import List, Literal
from .base_progress_tracker import BaseProgressTracker, ProgressInfo


class TrainerProgressTracker(BaseProgressTracker):
    """Progress tracker for BaseTrainer runs."""
    
    def get_runner_type(self) -> Literal['trainer']:
        return 'trainer'
    
    def get_expected_files(self) -> List[str]:
        return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    
    def get_log_pattern(self) -> str:
        return "train_val*.log"
    
    def calculate_progress(self) -> ProgressInfo:
        """Calculate trainer-specific progress using moved session_progress logic."""
        # Use moved session_progress logic (now in same module)
        from .session_progress import get_session_progress
        
        # Get basic progress info (preserves existing logic)  
        basic_progress = get_session_progress(self.work_dir, self.get_expected_files())
        
        # Enhance with new fields
        total_epochs = self.config.get('epochs') if self.config else None
        
        return ProgressInfo(
            # Existing fields (backward compatibility)
            completed_epochs=basic_progress.completed_epochs,
            progress_percentage=basic_progress.progress_percentage,
            early_stopped=basic_progress.early_stopped,
            early_stopped_at_epoch=basic_progress.early_stopped_at_epoch,
            
            # New fields
            runner_type='trainer',
            total_epochs=total_epochs,
        )
    
    def is_complete(self) -> bool:
        """Check if trainer is complete."""
        if not self.config or 'epochs' not in self.config:
            return False
        return self.get_progress().completed_epochs >= self.config['epochs']