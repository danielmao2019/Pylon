"""Gaussian Splatting operations and utilities."""

import importlib

# Import via importlib to avoid hard package deps and support Pylon packaging
render_rgb_from_gs = importlib.import_module(
    'utils.3dgs_ops.rendering'
).render_rgb_from_3dgs
apply_transform_to_gs = importlib.import_module(
    'utils.3dgs_ops.apply_transform'
).apply_transform_to_gs


__all__ = [
    'render_rgb_from_gs',
    'apply_transform_to_gs',
]
