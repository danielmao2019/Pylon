from typing import Any, Dict

SIDE_PANEL_STYLE: Dict[str, Any] = {
    "width": "20%",
    "padding": "20px",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    "height": "100vh",
    "position": "fixed",
    "left": 0,
    "top": 0,
}
MAIN_PANEL_STYLE: Dict[str, Any] = {
    "width": "80%",
    "marginLeft": "20%",
    "height": "100vh",
    "position": "relative",
    "overflow": "hidden",
}
PAGE_STYLE: Dict[str, Any] = {"margin": 0, "padding": 0}
LAYOUT_WRAPPER_STYLE: Dict[str, Any] = {
    "display": "flex",
    "width": "100%",
    "height": "100vh",
}
KEYBOARD_STYLE: Dict[str, Any] = {
    "width": "100%",
    "height": "100%",
    "position": "absolute",
    "top": 0,
    "left": 0,
    "pointerEvents": "none",
}
SLIDER_BLOCK_STYLE: Dict[str, Any] = {"margin-bottom": "30px"}
SLIDER_ROW_STYLE: Dict[str, Any] = {"display": "flex", "alignItems": "flex-start"}
SLIDER_COMPONENT_STYLE: Dict[str, Any] = {
    "flex": "1",
    "paddingBottom": "28px",
}
SLIDER_VALUE_STYLE: Dict[str, Any] = {
    "margin-left": "12px",
    "fontFamily": "monospace",
    "fontSize": "12px",
    "whiteSpace": "nowrap",
}
CAMERA_SELECTOR_SCROLLABLE_STYLE: Dict[str, Any] = {
    "maxHeight": "240px",
    "overflowY": "auto",
    "padding": "8px 12px",
    "border": "1px solid #dee2e6",
    "borderRadius": "4px",
    "backgroundColor": "#ffffff",
}
