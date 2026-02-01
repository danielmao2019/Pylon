"""Camera state management utilities for the viewer."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

import dash


def update_figures_parallel(
    figures: List[Dict[str, Any]],
    update_func: Callable[[int, Dict[str, Any]], Any],
    max_workers: int = 4
) -> List[Any]:
    """Update multiple figures in parallel using a given update function.

    Args:
        figures: List of figure dictionaries to update
        update_func: Function that takes (index, figure) and returns updated figure
        max_workers: Maximum number of worker threads

    Returns:
        List of updated figures
    """
    # Input validations
    assert isinstance(figures, list), f"{type(figures)=}"
    assert all(isinstance(figure, dict) for figure in figures), f"{figures=}"
    assert callable(update_func), f"{type(update_func)=}"
    assert isinstance(max_workers, int), f"{type(max_workers)=}"
    assert max_workers > 0, f"{max_workers=}"

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
    # Input validations
    assert isinstance(triggered_index, int), f"{type(triggered_index)=}"
    assert triggered_index >= 0, f"{triggered_index=}"
    assert isinstance(new_camera, dict), f"{type(new_camera)=}"

    def update_func(i: int, figure: Dict[str, Any]) -> Dict[str, Any]:
        # Input validations
        assert isinstance(i, int), f"{type(i)=}"
        assert isinstance(figure, dict), f"{type(figure)=}"
        assert "layout" in figure, f"{figure.keys()=}"
        assert "scene" in figure["layout"], f"{figure['layout'].keys()=}"

        if i == triggered_index:
            # Don't update the triggering graph
            return dash.no_update

        # Create updated figure with new camera state
        updated_figure = figure.copy()
        updated_figure["layout"] = updated_figure["layout"].copy()
        updated_figure["layout"]["scene"] = updated_figure["layout"]["scene"].copy()
        updated_figure["layout"]["scene"]["camera"] = new_camera
        return updated_figure

    return update_func


def reset_figure_camera() -> Callable[[int, Dict[str, Any]], Dict[str, Any]]:
    """Create a function to reset a figure's camera to Plotly's auto-calculated state.

    Returns:
        Function that resets a figure's camera state by removing manual camera
    """
    def reset_func(i: int, figure: Dict[str, Any]) -> Dict[str, Any]:
        # Input validations
        assert isinstance(i, int), f"{type(i)=}"
        assert isinstance(figure, dict), f"{type(figure)=}"
        assert "layout" in figure, f"{figure.keys()=}"
        assert "scene" in figure["layout"], f"{figure['layout'].keys()=}"

        updated_figure = figure.copy()
        updated_figure["layout"] = updated_figure["layout"].copy()
        updated_figure["layout"]["scene"] = updated_figure["layout"]["scene"].copy()

        # Remove manual camera to let Plotly auto-calculate
        if "camera" in updated_figure["layout"]["scene"]:
            del updated_figure["layout"]["scene"]["camera"]

        return updated_figure

    return reset_func
