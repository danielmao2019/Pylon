"""3D settings-related callbacks for the viewer."""

from typing import Dict, List, Optional, Union

from data.viewer.utils.settings_config import ViewerSettings


def _build_3d_settings_payload(
    point_size: float,
    point_opacity: float,
    sym_diff_radius: Optional[float],
    lod_type: Optional[str],
    density_percentage: Optional[int],
) -> List[Dict[str, Union[str, int, float, bool]]]:
    # Input validations
    assert isinstance(
        point_size, (int, float)
    ), f"point_size must be numeric, got {type(point_size)}"
    assert isinstance(
        point_opacity, (int, float)
    ), f"point_opacity must be numeric, got {type(point_opacity)}"
    assert sym_diff_radius is None or isinstance(
        sym_diff_radius, (int, float)
    ), f"sym_diff_radius must be numeric or None, got {type(sym_diff_radius)}"
    assert lod_type is None or isinstance(
        lod_type, str
    ), f"lod_type must be str or None, got {type(lod_type)}"
    assert density_percentage is None or isinstance(
        density_percentage, int
    ), f"density_percentage must be int or None, got {type(density_percentage)}"

    raw_settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'sym_diff_radius': sym_diff_radius,
        'lod_type': lod_type,
        'density_percentage': density_percentage,
    }
    settings = ViewerSettings.validate_3d_settings(
        ViewerSettings.get_3d_settings_with_defaults(raw_settings)
    )
    return [settings]
