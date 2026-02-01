"""Common display utilities for the viewer."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from dash import dcc, html

from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.debug import display_debug_outputs


class DisplayStyles:
    """Centralized styling constants for display components."""

    # Grid layout styles
    GRID_ITEM_50 = {'width': '50%', 'display': 'inline-block'}
    GRID_ITEM_33 = {'width': '33%', 'display': 'inline-block'}
    GRID_ITEM_VERTICAL = {'display': 'inline-block', 'vertical-align': 'top'}

    # Grid layout styles with margin
    GRID_ITEM_48_MARGIN = {'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '2%'}
    GRID_ITEM_48_NO_MARGIN = {'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}

    # Flex layout styles
    FLEX_WRAP = {'display': 'flex', 'flex-wrap': 'wrap'}
    FLEX_ALIGN_START = {'display': 'flex', 'align-items': 'flex-start'}

    # Spacing styles
    MARGIN_TOP_20 = {'margin-top': '20px'}
    MARGIN_TOP_10 = {'margin-top': '10px'}

    # Metadata display styles
    METADATA_CONTAINER = {
        'background-color': '#f0f0f0',
        'padding': '10px',
        'max-height': '200px',
        'overflow-y': 'auto',
        'border-radius': '5px'
    }

    # Statistics styles
    STATS_CONTAINER = {
        'width': '33%',
        'display': 'inline-block',
        'vertical-align': 'top'
    }


class ParallelFigureCreator:
    """Utility class for creating figures in parallel."""

    def __init__(self, max_workers: int = 4, enable_timing: bool = False):
        """Initialize the parallel figure creator.

        Args:
            max_workers: Maximum number of worker threads
            enable_timing: Whether to enable performance timing
        """
        # Input validations
        assert isinstance(max_workers, int), f"{type(max_workers)=}"
        assert max_workers > 0, f"{max_workers=}"
        assert isinstance(enable_timing, bool), f"{type(enable_timing)=}"

        self.max_workers = max_workers
        self.enable_timing = enable_timing

    def create_figures_parallel(
        self,
        figure_tasks: List[Callable[[], Any]],
        context_name: str = "Figure Creation"
    ) -> List[Any]:
        """Create figures in parallel using ThreadPoolExecutor.

        Args:
            figure_tasks: List of callable functions that create figures
            context_name: Name for timing context

        Returns:
            List of created figures in the same order as input tasks
        """
        # Input validations
        assert isinstance(figure_tasks, list), f"{type(figure_tasks)=}"
        assert all(callable(task_func) for task_func in figure_tasks), (
            f"{figure_tasks=}"
        )
        assert isinstance(context_name, str), f"{type(context_name)=}"

        if self.enable_timing:
            start_time = time.time()

        figures = [None] * len(figure_tasks)

        with ThreadPoolExecutor(max_workers=min(len(figure_tasks), self.max_workers)) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(task_func): idx
                for idx, task_func in enumerate(figure_tasks)
            }

            # Collect results in order
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                figures[idx] = future.result()

        if self.enable_timing:
            elapsed_time = time.time() - start_time
            print(f"[{context_name}] Parallel figure creation took {elapsed_time:.4f}s")

        return figures


def create_metadata_display(meta_info: Dict[str, Any]) -> List[html.Div]:
    """Create standardized metadata display.

    Args:
        meta_info: Dictionary containing metadata

    Returns:
        List of HTML components for metadata display
    """
    # Input validations
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"

    if not meta_info:
        return []

    return [
        html.H4("Metadata:"),
        html.Pre(
            format_value(meta_info),
            style=DisplayStyles.METADATA_CONTAINER
        )
    ]


def create_debug_display(debug_outputs: Optional[Dict[str, Any]]) -> List[html.Div]:
    """Create standardized debug output display.

    Args:
        debug_outputs: Dictionary containing debug outputs

    Returns:
        List of HTML components for debug display
    """
    # Input validations
    assert debug_outputs is None or isinstance(debug_outputs, dict), (
        f"{type(debug_outputs)=}"
    )

    if not debug_outputs:
        return []

    return [display_debug_outputs(debug_outputs)]


def create_statistics_display(
    stats_data: List[Dict[str, Any]],
    titles: List[str],
    width_style: str = "33%"
) -> List[html.Div]:
    """Create standardized statistics display.

    Args:
        stats_data: List of statistics dictionaries
        titles: List of titles for each statistics section
        width_style: Width style for each statistics section

    Returns:
        List of HTML components for statistics display
    """
    # Input validations
    assert isinstance(stats_data, list), f"{type(stats_data)=}"
    assert all(isinstance(stats, dict) for stats in stats_data), f"{stats_data=}"
    assert isinstance(titles, list), f"{type(titles)=}"
    assert all(isinstance(title, str) for title in titles), f"{titles=}"
    assert isinstance(width_style, str), f"{type(width_style)=}"
    assert len(stats_data) == len(titles), f"{len(stats_data)=}, {len(titles)=}"

    components = []

    for stats, title in zip(stats_data, titles, strict=True):
        style = DisplayStyles.STATS_CONTAINER.copy()
        style['width'] = width_style

        # Create list items, handling both simple values and Dash components
        list_items = []
        for k, v in stats.items():
            if hasattr(v, '_namespace') and hasattr(v, '_type'):
                # This is a Dash component, render it directly
                list_items.append(html.Li([f"{k}: ", v]))
            else:
                # This is a simple value, convert to string
                list_items.append(html.Li(f"{k}: {v}"))

        components.append(
            html.Div([
                html.H4(f"{title}:"),
                html.Ul(list_items)
            ], style=style)
        )

    return components


def create_figure_grid(
    figures: List[Any],
    width_style: str = "50%",
    height_style: str = "400px",
    graph_id_prefix: str = "point-cloud-graph"
) -> List[html.Div]:
    """Create a grid of figures with consistent styling.

    Args:
        figures: List of plotly figures or html.Div components
        width_style: Width style for each grid item
        height_style: Height style for each figure
        graph_id_prefix: Prefix for graph component IDs

    Returns:
        List of HTML components for figure grid
    """
    # Input validations
    assert isinstance(figures, list), f"{type(figures)=}"
    assert isinstance(width_style, str), f"{type(width_style)=}"
    assert isinstance(height_style, str), f"{type(height_style)=}"
    assert isinstance(graph_id_prefix, str), f"{type(graph_id_prefix)=}"

    grid_items = []

    for i, fig in enumerate(figures):
        # Check if this is an empty div placeholder
        if isinstance(fig, html.Div):
            # Just add the div directly with consistent styling
            grid_items.append(
                html.Div([
                    fig  # Use the div as-is (empty placeholder)
                ], style={
                    'width': width_style,
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'max-height': height_style,
                    'overflow': 'hidden',
                    'box-sizing': 'border-box'
                })
            )
        else:
            # Normal figure - wrap in dcc.Graph
            grid_items.append(
                html.Div([
                    dcc.Graph(
                        id={'type': graph_id_prefix, 'index': i},
                        figure=fig,
                        style={
                            'height': height_style,
                            'width': '100%',
                            'max-height': height_style,
                            'overflow': 'hidden'
                        }
                    )
                ], style={
                    'width': width_style,
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'max-height': height_style,
                    'overflow': 'hidden',
                    'box-sizing': 'border-box'
                })
            )

    return grid_items


def create_standard_datapoint_layout(
    figure_components: List[html.Div],
    stats_components: List[html.Div],
    meta_info: Dict[str, Any],
    debug_outputs: Optional[Dict[str, Any]] = None
) -> html.Div:
    """Create a standardized datapoint layout.

    Args:
        figure_components: List of figure components
        stats_components: List of statistics components
        meta_info: Metadata dictionary
        debug_outputs: Optional debug outputs dictionary

    Returns:
        Complete HTML layout
    """
    # Input validations
    assert isinstance(figure_components, list), f"{type(figure_components)=}"
    assert isinstance(stats_components, list), f"{type(stats_components)=}"
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert debug_outputs is None or isinstance(debug_outputs, dict), (
        f"{type(debug_outputs)=}"
    )

    # Create metadata display
    meta_display = create_metadata_display(meta_info)

    # Create debug display
    debug_display = create_debug_display(debug_outputs)

    # Build layout
    layout_components = [
        # Figure displays
        html.Div(figure_components),

        # Info section
        html.Div([
            html.Div(stats_components),
            html.Div(meta_display, style=DisplayStyles.MARGIN_TOP_20)
        ], style=DisplayStyles.MARGIN_TOP_20)
    ]

    # Add debug section if present
    if debug_display:
        layout_components.append(
            html.Div(debug_display, style=DisplayStyles.MARGIN_TOP_20)
        )

    return html.Div(layout_components)


class PerformanceTimer:
    """Utility class for performance timing in display functions."""

    def __init__(self, context_name: str, enabled: bool = False):
        """Initialize performance timer.

        Args:
            context_name: Name for timing context
            enabled: Whether timing is enabled
        """
        # Input validations
        assert isinstance(context_name, str), f"{type(context_name)=}"
        assert isinstance(enabled, bool), f"{type(enabled)=}"

        self.context_name = context_name
        self.enabled = enabled
        self.start_time = None

    def start(self):
        """Start timing."""
        # Input validations
        assert isinstance(self.enabled, bool), f"{type(self.enabled)=}"

        if self.enabled:
            self.start_time = time.time()
            print(f"[{self.context_name}] Starting at {self.start_time:.4f}")

    def checkpoint(self, checkpoint_name: str):
        """Record a timing checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint
        """
        # Input validations
        assert isinstance(checkpoint_name, str), f"{type(checkpoint_name)=}"

        if self.enabled and self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"[{self.context_name}] {checkpoint_name} took {elapsed:.4f}s")

    def finish(self):
        """Finish timing and print total elapsed time."""
        # Input validations
        assert isinstance(self.enabled, bool), f"{type(self.enabled)=}"

        if self.enabled and self.start_time is not None:
            total_elapsed = time.time() - self.start_time
            print(f"[{self.context_name}] Total time: {total_elapsed:.4f}s")
            return total_elapsed
        return None
