"""Class distribution toggle callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any, Tuple
from dash import Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry

import logging
logger = logging.getLogger(__name__)


@callback(
    outputs=[
        Output({'type': 'class-dist-plot', 'index': ALL}, 'style'),
        Output({'type': 'class-dist-toggle', 'index': ALL}, 'children')
    ],
    inputs=[Input({'type': 'class-dist-toggle', 'index': ALL}, 'n_clicks')],
    states=[State({'type': 'class-dist-plot', 'index': ALL}, 'style')],
    group="class_distribution"
)
def toggle_class_distribution_plots(
    n_clicks_list: List[Optional[int]],
    current_styles: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Toggle visibility of class distribution plots.
    
    This callback handles toggling of multiple class distribution plots using pattern-matching
    callbacks. Each plot can be toggled independently.
    
    Args:
        n_clicks_list: List of n_clicks values for each toggle button
        current_styles: List of current styles for each plot container
        
    Returns:
        Tuple of (new_styles_list, new_button_texts_list)
        
    Raises:
        PreventUpdate: If no button has been clicked
    """
    # Check if any button was clicked
    if not any(clicks is not None and clicks > 0 for clicks in n_clicks_list):
        raise PreventUpdate
    
    # Simpler approach: find the button that was most recently clicked
    # by comparing n_clicks values and looking for changes
    try:
        if not ctx.triggered:
            raise PreventUpdate
        
        # Extract the triggered index from the prop_id
        import json
        triggered_prop_id = ctx.triggered[0]['prop_id']
        component_id_str = triggered_prop_id.split('.')[0]
        component_id = json.loads(component_id_str)
        triggered_index = component_id['index']
        
        # Find which button in our list corresponds to this index
        # Since pattern-matching callbacks preserve order, we can match by index
        button_index = None
        for i, clicks in enumerate(n_clicks_list):
            if clicks is not None and clicks > 0:
                button_index = i
                break
                
        if button_index is None:
            raise PreventUpdate
            
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
        logger.warning(f"Failed to parse triggered component, falling back to simple toggle: {e}")
        # Fallback: just toggle the first button that has clicks > 0
        button_index = None
        for i, clicks in enumerate(n_clicks_list):
            if clicks is not None and clicks > 0:
                button_index = i
                break
        
        if button_index is None:
            raise PreventUpdate
    
    # Create new styles and button texts
    new_styles = []
    new_button_texts = []
    
    for i, (clicks, current_style) in enumerate(zip(n_clicks_list, current_styles)):
        if i == button_index:
            # Toggle this specific plot
            is_hidden = current_style.get('display', 'none') == 'none'
            
            if is_hidden:
                # Show plot
                new_style = {**current_style, 'display': 'block'}
                new_button_text = "📊 Hide Plot"
            else:
                # Hide plot
                new_style = {**current_style, 'display': 'none'}
                new_button_text = "📊 Show Plot"
            
            new_styles.append(new_style)
            new_button_texts.append(new_button_text)
        else:
            # Keep other plots unchanged
            new_styles.append(current_style)
            # Determine current button text based on current style
            is_hidden = current_style.get('display', 'none') == 'none'
            current_button_text = "📊 Show Plot" if is_hidden else "📊 Hide Plot"
            new_button_texts.append(current_button_text)
    
    return new_styles, new_button_texts


def register_class_distribution_toggle(component_index: int) -> None:
    """Register IDs for a class distribution component.
    
    This function should be called when creating class distribution components
    to ensure they have the correct pattern-matching IDs.
    
    Args:
        component_index: Unique index for this class distribution component
        
    Returns:
        Tuple of (toggle_button_id, plot_container_id) for the component
    """
    toggle_button_id = {'type': 'class-dist-toggle', 'index': component_index}
    plot_container_id = {'type': 'class-dist-plot', 'index': component_index}
    
    return toggle_button_id, plot_container_id


# Global counter for generating unique component indices
_component_counter = 0


def get_next_component_index() -> int:
    """Get the next unique component index.
    
    Returns:
        Unique integer index for a new class distribution component
    """
    global _component_counter
    _component_counter += 1
    return _component_counter
