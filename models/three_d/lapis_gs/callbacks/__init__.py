from typing import Any

import dash

from models.three_d.lapis_gs.callbacks.debugger_toggle import (
    register_lapis_debugger_toggle,
)
from models.three_d.lapis_gs.callbacks.density_change import (
    register_lapis_density_change,
)
from models.three_d.lapis_gs.callbacks.rgb_change import register_lapis_rgb_change

__all__ = ['register_callbacks']


def register_callbacks(
    dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
) -> None:
    register_lapis_rgb_change(app=app, viewer=viewer)
    register_lapis_density_change(app=app, viewer=viewer)
    register_lapis_debugger_toggle(app=app, viewer=viewer)
