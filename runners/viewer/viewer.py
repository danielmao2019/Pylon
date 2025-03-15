from pathlib import Path
import dash

from runners.viewer.states.trainer import TrainingState
from runners.viewer.layout.main import create_main_layout
from runners.viewer.callbacks.navigation import register_navigation_callbacks
from utils.automation.cfg_log_conversion import get_work_dir, get_repo_root


class TrainerViewer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        # Modify work directory
        # Example: ./logs/benchmarks/xxx -> ./logs/viewer/xxx
        self.work_dir = self._modify_work_dir(self.config_path)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.state = TrainingState(self.work_dir)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = create_main_layout()
        
        # Register callbacks
        register_navigation_callbacks(self.app, self.state)
    
    def _modify_work_dir(self, config_path: Path) -> Path:
        """Modify work directory to use viewer subdirectory.
        
        Example:
            Input:  ./configs/reproduce/change_detection/xxx/config.py
            Output: ./logs/viewer/change_detection/xxx
        """
        # Get the corresponding work directory using repo root
        work_dir = Path(get_work_dir(str(config_path)))
        # Get relative path from repo root
        rel_path = work_dir.relative_to(get_repo_root() / 'logs')
        # Construct new path with 'viewer' as second directory
        return get_repo_root() / 'logs' / 'viewer' / rel_path
    
    def run(self, host: str = "localhost", port: int = 8050, debug: bool = True):
        """Start the Dash server."""
        self.app.run(host=host, port=port, debug=debug)
