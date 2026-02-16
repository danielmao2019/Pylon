import math
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
MODEL_STORE_CONTAINER_STYLE: Dict[str, Any] = {"display": "none"}
LABEL_BOLD_STYLE: Dict[str, Any] = {"fontWeight": "bold"}
CONTROL_HEADER_STYLE: Dict[str, Any] = {"margin-bottom": "12px"}
SLIDER_LABEL_STYLE: Dict[str, Any] = {"font-weight": "bold"}
PRIMARY_BUTTON_STYLE: Dict[str, Any] = {
    "width": "100%",
    "padding": "10px",
    "fontSize": "14px",
    "margin-bottom": "10px",
}
RECORD_STATUS_STYLE: Dict[str, Any] = {
    "text-align": "center",
    "color": "#666",
    "fontSize": "12px",
}
CAMERA_INFO_HEADER_STYLE: Dict[str, Any] = {"margin-bottom": "10px"}
CAMERA_INFO_TEXT_STYLE: Dict[str, Any] = {
    "fontSize": "11px",
    "fontFamily": "monospace",
    "lineHeight": "1.4",
    "whiteSpace": "pre-wrap",
}
CAMERA_INFO_CONTAINER_STYLE: Dict[str, Any] = {"margin-bottom": "30px"}
CAMERA_OVERLAY_CONTAINER_STYLE: Dict[str, Any] = {"margin-bottom": "20px"}
CAMERA_SELECTOR_HEADER_STYLE: Dict[str, Any] = {"margin-bottom": "10px"}
CAMERA_SELECTOR_INPUT_STYLE: Dict[str, Any] = {"marginRight": "8px"}
CAMERA_SELECTOR_LABEL_STYLE: Dict[str, Any] = {
    "display": "block",
    "marginBottom": "6px",
    "fontSize": "12px",
    "cursor": "pointer",
}
CAMERA_SELECTOR_SPLIT_TITLE_STYLE: Dict[str, Any] = {"margin": "12px 0 6px 0"}
CAMERA_SELECTOR_SPLIT_CONTAINER_STYLE: Dict[str, Any] = {"marginBottom": "20px"}
CAMERA_SELECTOR_CONTAINER_STYLE: Dict[str, Any] = {"margin-bottom": "30px"}
KEYBOARD_SHORTCUTS_HEADER_STYLE: Dict[str, Any] = {"margin-bottom": "10px"}
KEYBOARD_SHORTCUTS_LIST_STYLE: Dict[str, Any] = {
    "fontSize": "12px",
    "lineHeight": "1.8",
}
CAMERA_SELECTOR_SCROLLABLE_STYLE: Dict[str, Any] = {
    "maxHeight": "240px",
    "overflowY": "auto",
    "padding": "8px 12px",
    "border": "1px solid #dee2e6",
    "borderRadius": "4px",
    "backgroundColor": "#ffffff",
}


def _make_grid_style(method_count: int) -> Dict[str, Any]:
    # Input validations
    assert isinstance(
        method_count, int
    ), f"method_count must be int, got {type(method_count)}"
    assert method_count >= 0, f"method_count must be non-negative, got {method_count}"

    cols = (
        max(1, int(math.ceil(math.sqrt(max(1, method_count)))))
        if method_count > 0
        else 1
    )
    return {
        "display": "grid",
        "gridTemplateColumns": f"repeat({cols}, minmax(0, 1fr))",
        "gap": "20px",
        "width": "100%",
        "height": "100%",
        "padding": "20px",
        "boxSizing": "border-box",
        "overflowY": "auto",
    }
