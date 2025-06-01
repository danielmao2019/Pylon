from typing import List
import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from utils.data_loader import load_validation_scores, extract_metric_scores
from utils.visualization import create_score_map, create_score_map_figure


def register_callbacks(app: dash.Dash, log_dirs: List[str]):
    """
    Registers all callbacks for the app.
    
    Args:
        app: Dash application instance
        log_dirs: List of paths to log directories
    """
    # Create outputs for each run's score map
    outputs = [Output(f'score-map-{i}', 'children') for i in range(len(log_dirs))]
    
    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_score_maps(epoch: int, metric: str) -> List[dict]:
        """
        Updates the score maps based on selected epoch and metric.
        
        Args:
            epoch: Selected epoch index
            metric: Selected metric name
            log_dirs: List of paths to log directories
            
        Returns:
            figures: List of Plotly figure dictionaries for each run
        """
        if metric is None:
            raise PreventUpdate
        
        figures = []
        for i, log_dir in enumerate(log_dirs):
            # Load scores for this run and epoch
            scores = load_validation_scores(log_dir, epoch)
            
            # Extract scores for selected metric
            metric_scores = extract_metric_scores(scores, metric)
            
            # Create score map
            score_map = create_score_map(metric_scores)
            
            # Create figure
            run_name = log_dir.split('/')[-1]
            fig = create_score_map_figure(score_map, f"{run_name} - {metric}")
            
            # Add to figures list
            figures.append(dcc.Graph(figure=fig))
        
        return figures
