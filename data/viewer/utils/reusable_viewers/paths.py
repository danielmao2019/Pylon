"""Filesystem roots for reusable viewer assets."""

from pathlib import Path
from typing import List
from urllib.parse import urlencode

REUSABLE_VIEWERS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REUSABLE_VIEWERS_ROOT.parents[3]
PROJECT_ROOT = REPO_ROOT / "project"
OFFICE_HOURS_OUTPUT_ROOT = Path("/pub0/daniel/office_hours_research")
BENCHMARKS_ROOT = OFFICE_HOURS_OUTPUT_ROOT / "benchmarks"
RECONSTRUCTIONS_ROOT = BENCHMARKS_ROOT / "reconstructions"
TASK_OUTPUTS_ROOT = OFFICE_HOURS_OUTPUT_ROOT / "task_outputs"
ASSETS_ROOT = REUSABLE_VIEWERS_ROOT / "assets"
SHARED_ASSETS_DIR = ASSETS_ROOT
STATIC_ROOT = ASSETS_ROOT / "static"
SHARED_STATIC_DIR = STATIC_ROOT
STATIC_ASSET_ROUTE_PREFIX = "/shared-static"
SHARED_STATIC_ROUTE_PREFIX = STATIC_ASSET_ROUTE_PREFIX
POINTCLOUD_VIEWER_FILENAME = "pointcloud_viewer.html"
SPLAT_VIEWER_FILENAME = "splat_viewer.html"


def existing_allowed_file_roots() -> List[Path]:
    """Return existing roots that shared viewer routes may serve.

    Args:
        None.

    Returns:
        Existing absolute roots that contain viewer-consumed artifacts.
    """
    roots = [BENCHMARKS_ROOT, TASK_OUTPUTS_ROOT, PROJECT_ROOT]
    existing_roots = [root.resolve() for root in roots if root.exists()]
    assert existing_roots, "No shared viewer file roots exist. roots=%s" % roots
    return existing_roots


def build_static_asset_route_pattern() -> str:
    """Build the Flask route pattern for shared static assets.

    Args:
        None.

    Returns:
        Flask route pattern for shared static assets.
    """
    return f"{SHARED_STATIC_ROUTE_PREFIX}/<path:filename>"


def build_shared_asset_route_pattern() -> str:
    """Build the Flask route pattern for shared support assets.

    Args:
        None.

    Returns:
        Flask route pattern for shared support assets.
    """
    return "/shared-assets/<path:filename>"


def get_shared_static_asset_path(filename: str) -> Path:
    """Return the absolute path for one shared static asset.

    Args:
        filename: Shared static filename.

    Returns:
        Absolute static asset path.
    """
    assert isinstance(filename, str), (
        "Static filename must be a string. filename=%r" % filename
    )
    assert filename, "Static filename must be non-empty. filename=%r" % filename
    requested_path = Path(filename)
    assert not requested_path.is_absolute(), (
        "Static filename must not be absolute. filename=%r" % filename
    )
    static_root = STATIC_ROOT.resolve()
    asset_path = (static_root / requested_path).resolve()
    assert asset_path.is_relative_to(static_root), (
        "Static asset path must stay inside static root. "
        "filename=%r asset_path=%s static_root=%s"
        % (filename, asset_path, static_root)
    )
    return asset_path


def get_shared_asset_path(filename: str) -> Path:
    """Return the absolute path for one shared support asset.

    Args:
        filename: Shared asset filename.

    Returns:
        Absolute asset path.
    """
    assert isinstance(filename, str), (
        "Asset filename must be a string. filename=%r" % filename
    )
    assert filename, "Asset filename must be non-empty. filename=%r" % filename
    requested_path = Path(filename)
    assert not requested_path.is_absolute(), (
        "Asset filename must not be absolute. filename=%r" % filename
    )
    assets_root = ASSETS_ROOT.resolve()
    asset_path = (assets_root / requested_path).resolve()
    assert asset_path.is_relative_to(assets_root), (
        "Asset path must stay inside assets root. "
        "filename=%r asset_path=%s assets_root=%s"
        % (filename, asset_path, assets_root)
    )
    return asset_path


def build_static_asset_src(
    filename: str,
    route_prefix: str = SHARED_STATIC_ROUTE_PREFIX,
) -> str:
    """Build one shared static URL for a viewer iframe asset.

    Args:
        filename: Shared static filename.
        route_prefix: Static route prefix to use for this viewer.

    Returns:
        URL path to the static asset.
    """
    assert isinstance(filename, str), (
        "Static filename must be a string. filename=%r" % filename
    )
    assert filename, "Static filename must be non-empty. filename=%r" % filename
    assert isinstance(route_prefix, str), (
        "Route prefix must be a string. route_prefix=%r" % route_prefix
    )
    assert route_prefix, (
        "Route prefix must be non-empty. route_prefix=%r" % route_prefix
    )
    return f"{route_prefix.rstrip('/')}/{filename}"


def build_pointcloud_viewer_src() -> str:
    """Build the point-cloud viewer static URL.

    Args:
        None.

    Returns:
        URL path to the point-cloud viewer bootstrap document.
    """
    return build_static_asset_src(filename=POINTCLOUD_VIEWER_FILENAME)


def build_splat_viewer_src() -> str:
    """Build the splat viewer static URL.

    Args:
        None.

    Returns:
        URL path to the splat viewer bootstrap document.
    """
    return build_static_asset_src(filename=SPLAT_VIEWER_FILENAME)


def build_pointcloud_viewer_url(file_path: Path, label: str) -> str:
    """Build a point-cloud viewer URL with autoload query parameters.

    Args:
        file_path: PLY path to load in the shared viewer.
        label: Human-readable viewer label.

    Returns:
        URL path to the point-cloud viewer document.
    """
    assert isinstance(file_path, Path), (
        "File path must be a Path. file_path=%r" % file_path
    )
    assert isinstance(label, str), "Label must be a string. label=%r" % label
    query = urlencode({"file": str(file_path), "label": label})
    return f"{build_pointcloud_viewer_src()}?{query}"
