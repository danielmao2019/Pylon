"""Style constants for the texture-extraction benchmark viewer."""

from typing import Dict

ROOT_STYLE: Dict[str, str] = {
    "display": "flex",
    "minHeight": "100vh",
    "backgroundColor": "#edf2f7",
    "fontFamily": "Georgia, 'Times New Roman', serif",
    "color": "#102a43",
}

LEFT_PANEL_STYLE: Dict[str, str] = {
    "width": "280px",
    "padding": "24px 20px",
    "background": "linear-gradient(180deg, #102a43 0%, #243b53 100%)",
    "color": "#f0f4f8",
    "boxShadow": "inset -1px 0 0 rgba(255, 255, 255, 0.08)",
}

LEFT_PANEL_CARD_STYLE: Dict[str, str] = {
    "padding": "16px",
    "borderRadius": "14px",
    "backgroundColor": "rgba(255, 255, 255, 0.08)",
    "border": "1px solid rgba(255, 255, 255, 0.12)",
}

RIGHT_PANEL_STYLE: Dict[str, str] = {
    "flex": "1",
    "padding": "24px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "18px",
}

ROW_TWO_STYLE: Dict[str, str] = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "16px",
}

ROW_THREE_STYLE: Dict[str, str] = {
    "display": "grid",
    "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
    "gap": "16px",
}

CARD_STYLE: Dict[str, str] = {
    "backgroundColor": "#ffffff",
    "borderRadius": "16px",
    "padding": "14px",
    "boxShadow": "0 10px 30px rgba(15, 23, 42, 0.08)",
    "display": "flex",
    "flexDirection": "column",
    "gap": "12px",
    "minHeight": "280px",
}

GRAPH_STYLE: Dict[str, str] = {
    "height": "100%",
}

MESH_CARD_STYLE: Dict[str, str] = {
    **CARD_STYLE,
    "minHeight": "360px",
}

MESH_WRAPPER_STYLE: Dict[str, str] = {
    "height": "320px",
}

TAB_STYLE: Dict[str, str] = {
    "padding": "12px 18px",
    "fontWeight": "600",
    "backgroundColor": "#d9e2ec",
    "border": "none",
}

TAB_SELECTED_STYLE: Dict[str, str] = {
    **TAB_STYLE,
    "backgroundColor": "#ffffff",
    "color": "#102a43",
    "borderBottom": "3px solid #d64545",
}
