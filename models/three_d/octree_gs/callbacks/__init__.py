from typing import Any

import dash

from models.three_d.octree_gs.callbacks.debugger_toggle import (
    register_octree_debugger_toggle,
)
from models.three_d.octree_gs.callbacks.density_change import (
    register_octree_density_change,
)
from models.three_d.octree_gs.callbacks.rgb_change import register_octree_rgb_change

__all__ = ['register_callbacks']


def register_callbacks(
    dataset: Any, app: dash.Dash, viewer: Any, **kwargs: Any
) -> None:
    register_octree_rgb_change(app=app, viewer=viewer)
    register_octree_density_change(app=app, viewer=viewer)
    register_octree_debugger_toggle(app=app, viewer=viewer)
