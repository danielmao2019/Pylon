"""Module-level runtime state for the texture-extraction benchmark viewer."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.models.three_d.meshes.texture.extract.viewer.backend.benchmark_backend import (
    load_results_index,
    load_scene_payload,
)

_RESULTS_ROOT: Optional[Path] = None
_RESULTS_INDEX: Optional[Dict[str, Any]] = None


def configure_viewer(
    results_root: Path,
) -> None:
    """Configure the module-level viewer state.

    Args:
        results_root: Benchmark results root.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(results_root, Path), (
            "Expected `results_root` to be a `Path`. " f"{type(results_root)=}."
        )

    _validate_inputs()

    global _RESULTS_ROOT
    global _RESULTS_INDEX
    _RESULTS_ROOT = results_root
    _RESULTS_INDEX = load_results_index(results_root=results_root)


def get_results_root() -> Path:
    """Return the configured benchmark results root.

    Args:
        None.

    Returns:
        Configured results root.
    """

    assert _RESULTS_ROOT is not None, (
        "Expected the benchmark viewer state to be configured before use. "
        f"{_RESULTS_ROOT=}"
    )
    return _RESULTS_ROOT


def get_results_index() -> Dict[str, Any]:
    """Return the configured benchmark results index.

    Args:
        None.

    Returns:
        Cached results-index dictionary.
    """

    assert _RESULTS_INDEX is not None, (
        "Expected the benchmark viewer state to be configured before use. "
        f"{_RESULTS_INDEX=}"
    )
    return _RESULTS_INDEX


def get_scene_names() -> List[str]:
    """Return the configured benchmark scene names.

    Args:
        None.

    Returns:
        Scene-name list.
    """

    results_index = get_results_index()
    scene_names = results_index["scene_names"]
    assert isinstance(scene_names, list), (
        "Expected `scene_names` to be a list. " f"{type(scene_names)=}."
    )
    return scene_names


def get_default_scene_name() -> str:
    """Return the configured default scene name.

    Args:
        None.

    Returns:
        Default scene name.
    """

    results_index = get_results_index()
    default_scene_name = results_index["default_scene_name"]
    assert isinstance(default_scene_name, str), (
        "Expected `default_scene_name` to be a string. " f"{type(default_scene_name)=}."
    )
    return default_scene_name


def get_open3d_gpu_supported() -> bool:
    """Return whether the cached benchmark includes an Open3D GPU row.

    Args:
        None.

    Returns:
        Open3D-GPU support flag.
    """

    results_index = get_results_index()
    open3d_gpu_supported = results_index["open3d_gpu_supported"]
    assert isinstance(open3d_gpu_supported, bool), (
        "Expected `open3d_gpu_supported` to be a bool. "
        f"{type(open3d_gpu_supported)=}."
    )
    return open3d_gpu_supported


def get_scene_payload(
    scene_name: str,
) -> Dict[str, Any]:
    """Load one scene payload from the configured results root.

    Args:
        scene_name: Benchmark scene name.

    Returns:
        Scene payload dictionary.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(scene_name, str), (
            "Expected `scene_name` to be a string. " f"{type(scene_name)=}."
        )
        assert scene_name != "", (
            "Expected `scene_name` to be non-empty. " f"{scene_name=}"
        )

    _validate_inputs()

    return load_scene_payload(
        scene_name=scene_name,
        results_root=get_results_root(),
    )
