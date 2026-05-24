"""File resolution helpers for shared viewer artifact serving."""

from pathlib import Path
from typing import List, Optional

from data.viewer.utils.reusable_viewers.paths import existing_allowed_file_roots


def is_relative_to(path: Path, root: Path) -> bool:
    """Check whether a path is contained by a root.

    Args:
        path: Absolute candidate path.
        root: Absolute allowed root path.

    Returns:
        Whether `path` is contained by `root`.
    """
    assert path.is_absolute(), "Candidate path must be absolute. path=%s" % path
    assert root.is_absolute(), "Root path must be absolute. root=%s" % root
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_allowed_file(path_value: str) -> Path:
    """Resolve a route path into an allowed existing file.

    Args:
        path_value: Absolute artifact path from a viewer URL query parameter.

    Returns:
        Resolved existing file path.
    """
    assert isinstance(path_value, str), "Path value must be a string. path_value=%r" % (
        path_value,
    )
    assert path_value, "Path value must be non-empty. path_value=%r" % path_value
    resolved_path = Path(path_value).expanduser().resolve()
    assert resolved_path.exists(), "Shared viewer file is missing. path=%s" % (
        resolved_path,
    )
    assert resolved_path.is_file(), "Shared viewer path is not a file. path=%s" % (
        resolved_path,
    )
    allowed_roots = existing_allowed_file_roots()
    is_allowed = any(is_relative_to(resolved_path, root) for root in allowed_roots)
    assert is_allowed, "Shared viewer file is outside allowed roots. path=%s roots=%s" % (
        resolved_path,
        allowed_roots,
    )
    return resolved_path


def first_existing_file(candidates: List[Path]) -> Optional[Path]:
    """Return the first existing file from ordered candidates.

    Args:
        candidates: Ordered candidate artifact paths.

    Returns:
        First existing file, or `None` when no candidate exists.
    """
    assert isinstance(candidates, list), "Candidates must be a list. candidates=%r" % (
        candidates,
    )
    for candidate in candidates:
        assert isinstance(candidate, Path), "Candidate must be a Path. candidate=%r" % (
            candidate,
        )
        if candidate.is_file():
            return candidate
    return None


def first_ply_under(root: Path) -> Optional[Path]:
    """Return the first displayable PLY artifact under a root.

    Args:
        root: Directory that may contain point-cloud artifacts.

    Returns:
        First non-camera PLY file, or `None` when no displayable PLY exists.
    """
    assert isinstance(root, Path), "Root must be a Path. root=%r" % (root,)
    if not root.exists():
        return None
    assert root.is_dir(), "Root must be a directory when it exists. root=%s" % root
    preferred_names = [
        "combined_pcd.ply",
        "input.ply",
        "model.ply",
        "point_cloud.ply",
    ]
    for preferred_name in preferred_names:
        matches = sorted(root.rglob(preferred_name))
        for match in matches:
            if match.is_file() and match.name != "camera_poses.ply":
                return match
    for match in sorted(root.rglob("*.ply")):
        if match.is_file() and match.name != "camera_poses.ply":
            return match
    return None
