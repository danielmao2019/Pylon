"""Camera state management utilities for the viewer."""
from typing import Dict, List, Any, Callable
import dash
from concurrent.futures import ThreadPoolExecutor, as_completed


def update_figures_parallel(
    figures: List[Dict[str, Any]],
    update_func: Callable[[int, Dict[str, Any]], Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """Update multiple figures in parallel using a given update function.
    
    Args:
        figures: List of figure dictionaries to update
        update_func: Function that takes (index, figure) and returns updated figure
        max_workers: Maximum number of worker threads
        
    Returns:
        List of updated figures
    """
    updated_figures = [None] * len(figures)
    
    # Use parallel processing for multiple figures
    if len(figures) > 1:
        with ThreadPoolExecutor(max_workers=min(len(figures), max_workers)) as executor:
            # Submit all update tasks
            future_to_index = {
                executor.submit(update_func, i, figure): i 
                for i, figure in enumerate(figures)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                updated_figures[idx] = future.result()
    else:
        # For single figure, just update directly
        updated_figures = [update_func(i, figure) for i, figure in enumerate(figures)]
    
    return updated_figures


def update_figure_camera(triggered_index: int, new_camera: Dict[str, Any]) -> Callable[[int, Dict[str, Any]], Dict[str, Any]]:
    """Create a function to update a figure's camera state.
    
    Args:
        triggered_index: Index of the figure that triggered the update (skip this one)
        new_camera: New camera state to apply
        
    Returns:
        Function that updates a figure's camera state
    """
    def update_func(i: int, figure: Dict[str, Any]) -> Dict[str, Any]:
        if i == triggered_index or not figure:
            # Don't update the triggering graph or empty figures
            return dash.no_update
        else:
            # Create updated figure with new camera state
            updated_figure = figure.copy()
            if 'layout' not in updated_figure:
                updated_figure['layout'] = {}
            if 'scene' not in updated_figure['layout']:
                updated_figure['layout']['scene'] = {}
            updated_figure['layout']['scene']['camera'] = new_camera
            return updated_figure
    
    return update_func


def reset_figure_camera() -> Callable[[int, Dict[str, Any]], Dict[str, Any]]:
    """Create a function to reset a figure's camera to Plotly's auto-calculated state.
    
    Returns:
        Function that resets a figure's camera state by removing manual camera
    """
    def reset_func(i: int, figure: Dict[str, Any]) -> Dict[str, Any]:
        if not figure:
            return dash.no_update

        updated_figure = figure.copy()
        if 'layout' not in updated_figure:
            updated_figure['layout'] = {}
        if 'scene' not in updated_figure['layout']:
            updated_figure['layout']['scene'] = {}
        
        # Remove manual camera to let Plotly auto-calculate
        if 'camera' in updated_figure['layout']['scene']:
            del updated_figure['layout']['scene']['camera']
        
        return updated_figure
    
    return reset_func
