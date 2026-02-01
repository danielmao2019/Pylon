"""Centralized configuration for viewer settings."""
from typing import Dict, Optional, Union


class ViewerSettings:
    """Centralized configuration for viewer settings."""

    # Default 3D visualization settings
    DEFAULT_3D_SETTINGS = {
        'point_size': 2.0,
        'point_opacity': 0.8,
        'sym_diff_radius': 0.05,
        'lod_type': 'none',
        'density_percentage': 100
    }

    # LOD type options
    LOD_TYPE_OPTIONS = [
        {'label': 'No LOD', 'value': 'none'},
        {'label': 'Continuous LOD (Adaptive)', 'value': 'continuous'},
        {'label': 'Discrete LOD (Pre-computed)', 'value': 'discrete'}
    ]

    # Performance settings
    PERFORMANCE_SETTINGS = {
        'max_thread_workers': 4,
        'camera_update_debounce': 100  # milliseconds
    }

    @classmethod
    def get_3d_settings_with_defaults(
        cls,
        settings_3d: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> Dict[str, Union[float, str]]:
        """Extract 3D settings with proper defaults and validation.

        Args:
            settings_3d: Raw 3D settings dictionary from store

        Returns:
            Validated 3D settings with defaults applied
        """
        # Input validations
        assert settings_3d is None or isinstance(settings_3d, dict), (
            f"{type(settings_3d)=}"
        )
        assert settings_3d is None or all(
            isinstance(key, str) for key in settings_3d
        ), f"{settings_3d=}"
        assert settings_3d is None or not (
            set(settings_3d.keys()) - set(cls.DEFAULT_3D_SETTINGS.keys())
        ), f"{settings_3d=}"

        # Input normalizations
        if settings_3d is None:
            settings_3d = {}

        result = cls.DEFAULT_3D_SETTINGS.copy()
        result.update(settings_3d)
        return result

    @classmethod
    def validate_3d_settings(
        cls, settings: Dict[str, Union[str, int, float, bool]]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Validate 3D settings to acceptable ranges.

        Args:
            settings: Raw settings dictionary

        Returns:
            Validated settings
        """
        # Input validations
        assert isinstance(settings, dict), f"{type(settings)=}"
        assert all(isinstance(key, str) for key in settings), f"{settings=}"
        assert not (set(cls.DEFAULT_3D_SETTINGS.keys()) - set(settings.keys())), (
            f"{settings=}"
        )
        assert isinstance(settings["point_size"], (int, float)), (
            f"{type(settings['point_size'])=}"
        )
        assert isinstance(settings["point_opacity"], (int, float)), (
            f"{type(settings['point_opacity'])=}"
        )
        assert isinstance(settings["sym_diff_radius"], (int, float)), (
            f"{type(settings['sym_diff_radius'])=}"
        )
        assert isinstance(settings["density_percentage"], int), (
            f"{type(settings['density_percentage'])=}"
        )
        assert isinstance(settings["lod_type"], str), f"{type(settings['lod_type'])=}"
        assert 0.1 <= settings["point_size"] <= 20.0, f"{settings['point_size']=}"
        assert 0.0 <= settings["point_opacity"] <= 1.0, (
            f"{settings['point_opacity']=}"
        )
        assert 0.0 <= settings["sym_diff_radius"] <= 2.0, (
            f"{settings['sym_diff_radius']=}"
        )
        assert 10 <= settings["density_percentage"] <= 100, (
            f"{settings['density_percentage']=}"
        )
        assert settings["density_percentage"] % 10 == 0, (
            f"{settings['density_percentage']=}"
        )
        assert settings["lod_type"] in {
            opt["value"] for opt in cls.LOD_TYPE_OPTIONS
        }, f"{settings['lod_type']=}"

        return settings
