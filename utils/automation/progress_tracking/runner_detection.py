from typing import Dict, Any, Optional, Literal
import os


def detect_runner_type(work_dir: str, config: Optional[Dict[str, Any]] = None) -> Literal['trainer', 'evaluator']:
    """Detect runner type from work_dir structure or config. 
    
    This utility is shared between progress_tracking and eval_viewer.
    
    Args:
        work_dir: Path to log/work directory
        config: Optional config dictionary for additional context
        
    Returns:
        'trainer' if directory contains BaseTrainer results
        'evaluator' if directory contains BaseEvaluator results
        
    Raises:
        ValueError: If runner type cannot be determined (FAIL FAST)
    """
    # Strategy 1: Check existing files (based on eval_viewer's proven approach)
    # Check for BaseEvaluator pattern: evaluation_scores.json directly in work_dir
    if os.path.exists(os.path.join(work_dir, "evaluation_scores.json")):
        return 'evaluator'
    
    # Check for BaseTrainer pattern: epoch folders with validation_scores.json
    epoch_0_dir = os.path.join(work_dir, "epoch_0")
    validation_scores_path = os.path.join(epoch_0_dir, "validation_scores.json")
    if os.path.exists(epoch_0_dir) and os.path.exists(validation_scores_path):
        return 'trainer'
    
    # Strategy 2: Check config if available (additional context)
    if config:
        assert 'runner' in config, f"Config must have 'runner' key, got keys: {list(config.keys())}"
        runner_config = config['runner']
        # Enforce contract: runner config must be a direct class reference
        assert isinstance(runner_config, type), f"Expected runner to be a class, got {type(runner_config)}: {runner_config}"
        
        class_name = runner_config.__name__ if hasattr(runner_config, '__name__') else str(runner_config)
        if 'Evaluator' in class_name:
            return 'evaluator' 
        elif 'Trainer' in class_name:
            return 'trainer'
                
        # Strategy 3: Check for 'epochs' field (trainers have this)
        if 'epochs' in config:
            return 'trainer'
    
    # FAIL FAST: Cannot determine runner type
    available_files = os.listdir(work_dir) if os.path.exists(work_dir) else []
    config_info = f"Config keys: {list(config.keys())}" if config else "No config provided"
    
    raise ValueError(
        f"Unable to detect runner type for work_dir: {work_dir}\n"
        f"Available files: {available_files}\n"
        f"{config_info}\n"
        f"Expected patterns:\n"
        f"  - Trainer: epoch_0/ directory with validation_scores.json OR 'epochs' in config\n"
        f"  - Evaluator: evaluation_scores.json file OR 'Evaluator' in runner class name"
    )
