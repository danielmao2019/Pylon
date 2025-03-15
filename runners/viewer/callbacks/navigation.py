from dash import Input, Output, ctx
import plotly.graph_objects as go


def register_navigation_callbacks(app, state):
    @app.callback(
        [Output("iteration-display", "children"),
         Output("sample-display", "children"),
         Output("input-image-1", "figure"),
         Output("input-image-2", "figure"),
         Output("pred-change-map", "figure"),
         Output("gt-change-map", "figure")],
        [Input("btn-prev-iter", "n_clicks"),
         Input("btn-next-iter", "n_clicks"),
         Input("btn-prev-sample", "n_clicks"),
         Input("btn-next-sample", "n_clicks")]
    )
    def update_display(prev_iter_clicks, next_iter_clicks, prev_sample_clicks, next_sample_clicks):
        # Determine which button was clicked using the new ctx module
        if not ctx.triggered:
            button_id = None
        else:
            button_id = ctx.triggered_id
        
        # Update based on button click
        if button_id == "btn-next-iter":
            state.next_iteration()
        elif button_id == "btn-prev-iter":
            state.prev_iteration()
        elif button_id == "btn-next-sample":
            state.next_sample()
        elif button_id == "btn-prev-sample":
            state.prev_sample()
        
        # Get current data
        data = state.get_current_data()
        nav_info = state.get_navigation_info()
        
        # Create figures
        input1_fig = create_image_figure(data["input1"].transpose(1, 2, 0))
        input2_fig = create_image_figure(data["input2"].transpose(1, 2, 0))
        
        # Convert predictions and ground truth to RGB
        pred_rgb = state.class_to_rgb(data["pred"])
        gt_rgb = state.class_to_rgb(data["gt"])
        
        pred_fig = create_image_figure(pred_rgb / 255.0)
        gt_fig = create_image_figure(gt_rgb / 255.0)
        
        return (
            f"Training Iteration: {nav_info['current_iteration']} / {nav_info['total_iterations']}",
            f"Sample: {nav_info['current_sample']} / {nav_info['batch_size']}",
            input1_fig,
            input2_fig,
            pred_fig,
            gt_fig
        )

def create_image_figure(img_array):
    """Create a plotly figure from an image array."""
    return {
        "data": [
            go.Image(
                z=img_array,
                hoverinfo="none"
            )
        ],
        "layout": go.Layout(
            xaxis={"showticklabels": False, "showgrid": False},
            yaxis={"showticklabels": False, "showgrid": False},
            margin=dict(l=0, r=0, t=0, b=0),
            dragmode=False
        )
    }
