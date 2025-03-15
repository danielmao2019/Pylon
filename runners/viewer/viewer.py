from pathlib import Path
import dash

from runners.viewer.states.trainer import TrainingState
from runners.viewer.layout.main import create_main_layout
from runners.viewer.callbacks.navigation import register_navigation_callbacks


class TrainerViewer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        
        # Initialize state
        self.state = TrainingState(self.config_path)
        
        # Create work directory
        work_dir = Path(self.state.config['work_dir'])
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.layout = create_main_layout()
        
        # Register callbacks
        register_navigation_callbacks(self.app, self.state)
    
    def run(self, host: str = "localhost", port: int = 8050, debug: bool = True):
        """Start the Dash server."""
        self.app.run(host=host, port=port, debug=debug)
