from pathlib import Path
import dash

from runners.viewer.layout.main import create_main_layout
from runners.viewer.states.trainer import TrainingState
from runners.viewer.callbacks.navigation import register_navigation_callbacks


class TrainerViewer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        # Modify work directory
        # Example: ./logs/benchmarks/xxx -> ./logs/viewer/xxx
        self.work_dir = self._modify_work_dir(self.config_path)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state
        self.state = TrainingState()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = create_main_layout()
        
        # Register callbacks
        register_navigation_callbacks(self.app, self.state)
    
    def _modify_work_dir(self, config_path: Path) -> Path:
        # TODO: Implement proper config parsing
        # For now, just create a dummy path
        return Path("./logs/viewer/test")
    
    def run(self, host: str = "localhost", port: int = 8050, debug: bool = True):
        """Start the Dash server."""
        self.app.run(host=host, port=port, debug=debug)
