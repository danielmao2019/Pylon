"""Styles for the evaluation viewer layout."""

APP_TITLE_STYLE = {"textAlign": "center"}

CONTROLS_CONTAINER_STYLE = {"padding": "20px"}
CONTROL_SECTION_STYLE = {"width": "33%", "display": "inline-block"}
CONTROL_SECTION_RIGHT_STYLE = {
    "width": "33%",
    "display": "inline-block",
    "float": "right",
}

SECTION_TITLE_STYLE = {"textAlign": "center"}
AGGREGATED_PLOT_CONTAINER_STYLE = {"marginTop": "20px"}
AGGREGATED_PLOT_STYLE = {"width": "100%"}

COLOR_BAR_CONTAINER_STYLE = {"marginLeft": "10px"}
COLOR_BAR_ROW_STYLE = {"display": "flex", "alignItems": "center"}
COLOR_BAR_GRADIENT_STYLE = {
    "width": "20px",
    "height": "100px",
    "background": "linear-gradient(to bottom, rgb(0,255,0), rgb(255,255,0), rgb(255,0,0))",
    "marginRight": "10px",
}
COLOR_BAR_LABELS_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "space-between",
}

GRID_PADDING_STYLE = {
    "width": "20px",
    "height": "20px",
    "padding": "0",
    "margin": "0",
}
GRID_BUTTON_BASE_STYLE = {
    "width": "20px",
    "height": "20px",
    "padding": "0",
    "margin": "0",
    "border": "none",
}
GRID_BUTTON_NAN_STYLE = {
    **GRID_BUTTON_BASE_STYLE,
    "backgroundColor": "#f0f0f0",
    "cursor": "not-allowed",
}
GRID_BUTTON_VALID_STYLE = {
    **GRID_BUTTON_BASE_STYLE,
    "cursor": "pointer",
}
GRID_CONTAINER_BASE_STYLE = {
    "display": "grid",
    "gap": "1px",
    "width": "fit-content",
    "margin": "0 auto",
}

INDIVIDUAL_SCORE_MAPS_CONTAINER_STYLE = {"display": "flex", "flexWrap": "wrap"}
INDIVIDUAL_RUN_CONTAINER_STYLE = {"width": "50%", "display": "inline-block"}
INDIVIDUAL_RUN_TITLE_STYLE = {"textAlign": "center", "marginBottom": "10px"}
INDIVIDUAL_RUN_ROW_STYLE = {"display": "flex", "alignItems": "center"}
INDIVIDUAL_BUTTON_GRID_STYLE = {"width": "100%", "display": "inline-block"}
INDIVIDUAL_COLOR_BAR_STYLE = {"display": "inline-block"}

OVERLAID_SECTION_STYLE = {"marginTop": "20px"}
OVERLAID_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fill, minmax(20px, 1fr))",
    "gap": "1px",
    "width": "100%",
    "maxWidth": "800px",
    "margin": "0 auto",
}
OVERLAID_ROW_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
}

DATAPOINT_SECTION_STYLE = {"marginTop": "20px"}
DATAPOINT_DISPLAY_STYLE = {"width": "100%"}

LEFT_COLUMN_STYLE = {
    "width": "60%",
    "float": "left",
    "height": "calc(100vh - 100px)",
    "overflowY": "auto",
    "padding": "20px",
    "boxSizing": "border-box",
}
RIGHT_COLUMN_STYLE = {
    "width": "40%",
    "float": "right",
    "height": "calc(100vh - 100px)",
    "overflowY": "auto",
    "padding": "20px",
    "boxSizing": "border-box",
    "borderLeft": "1px solid #ddd",
}
MAIN_CONTENT_STYLE = {
    "display": "flex",
    "width": "100%",
    "height": "calc(100vh - 100px)",
    "position": "relative",
}
