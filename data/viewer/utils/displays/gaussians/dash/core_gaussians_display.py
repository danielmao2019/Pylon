"""Dash Gaussian-splat display object core.

The Dash Gaussian display tree is part of the data.viewer atomic-display
restructure. The benchmark viewer renders Gaussians through the TypeScript
backend/frontend pair under `gaussians/ts/`; no Dash caller in this branch
exercises the Dash Gaussian path, so the Dash entry points expose the
skeleton-declared signatures without a concrete renderer.
"""

from typing import Any, Dict, Optional


def create_dash_gaussians_display(
    gaussian_path: Optional[str],
    title: str,
    meta_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a Dash Gaussian-splat display object.

    Args:
        gaussian_path: Gaussian-splat artifact path.
        title: Display panel title.
        meta_info: Optional renderer metadata.

    Returns:
        Dash Gaussian-splat display component.
    """
    assert gaussian_path is None or isinstance(gaussian_path, str), (
        "Gaussian path must be None or a string. gaussian_path=%r" % gaussian_path
    )
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )
    raise NotImplementedError(
        "Dash Gaussian display is declared by the skeleton but not exercised "
        "by any caller in this branch. The benchmark viewer renders Gaussians "
        "through data.viewer.utils.displays.gaussians.ts."
    )


def create_dash_gaussians_scene(
    gaussian_path: Optional[str],
    title: str,
    meta_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a Dash Gaussian-splat display scene from Gaussian data and display metadata.

    Args:
        gaussian_path: Gaussian-splat artifact path.
        title: Display panel title.
        meta_info: Optional renderer metadata.

    Returns:
        Scene payload consumed by the Dash Gaussian component.
    """
    assert gaussian_path is None or isinstance(gaussian_path, str), (
        "Gaussian path must be None or a string. gaussian_path=%r" % gaussian_path
    )
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )
    raise NotImplementedError(
        "Dash Gaussian scene construction is declared by the skeleton but not "
        "exercised by any caller in this branch."
    )


def create_dash_gaussians_component(
    scene: Any,
    controls: Any,
    title: str,
) -> Any:
    """Wrap the Dash Gaussian-splat scene and camera controls into a Dash component.

    Args:
        scene: Gaussian display scene payload.
        controls: Dash trackball camera controls component.
        title: Display panel title.

    Returns:
        Dash Gaussian-splat display component.
    """
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    raise NotImplementedError(
        "Dash Gaussian component assembly is declared by the skeleton but not "
        "exercised by any caller in this branch."
    )
