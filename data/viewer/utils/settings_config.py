"""Centralized configuration for viewer settings."""
from typing import Dict, Union


class ViewerSettings:
    """Centralized configuration for viewer settings."""
    
    # Default 3D visualization settings
    DEFAULT_3D_SETTINGS = {
        'point_size': 2.0,
        'point_opacity': 0.8,
        'sym_diff_radius': 0.05,
        'corr_radius': 0.1,
        'lod_type': 'none',
        'density_percentage': 50
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
        settings_3d: Dict[str, Union[str, int, float, bool]] = None
    ) -> Dict[str, Union[float, str]]:
        """Extract 3D settings with proper defaults and validation.
        
        Args:
            settings_3d: Raw 3D settings dictionary from store
            
        Returns:
            Validated 3D settings with defaults applied
        """
        if settings_3d is None:
            settings_3d = {}
        
        # Apply defaults for missing values
        result = {}
        for key, default_value in cls.DEFAULT_3D_SETTINGS.items():
            result[key] = settings_3d.get(key, default_value)
        
        return result
    
    @classmethod
    def validate_3d_settings(
        cls,
        settings: Dict[str, Union[str, int, float, bool]]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Validate and clamp 3D settings to acceptable ranges.
        
        Args:
            settings: Raw settings dictionary
            
        Returns:
            Validated and clamped settings
        """
        validated = settings.copy()
        
        # Clamp numeric values to acceptable ranges
        validated['point_size'] = max(0.1, min(20.0, float(validated.get('point_size', 2.0))))
        validated['point_opacity'] = max(0.0, min(1.0, float(validated.get('point_opacity', 0.8))))
        validated['sym_diff_radius'] = max(0.0, min(2.0, float(validated.get('sym_diff_radius', 0.05))))
        validated['corr_radius'] = max(0.0, min(2.0, float(validated.get('corr_radius', 0.1))))
        validated['density_percentage'] = max(1, min(100, int(validated.get('density_percentage', 100))))
        
        # Validate LOD type
        valid_lod_types = {opt['value'] for opt in cls.LOD_TYPE_OPTIONS}
        if validated.get('lod_type') not in valid_lod_types:
            validated['lod_type'] = cls.DEFAULT_3D_SETTINGS['lod_type']
        
        return validated
    
    @classmethod
    def requires_3d_visualization(cls, dataset_type: str) -> bool:
        """Check if a dataset type requires 3D visualization.
        
        This function checks if the dataset type uses point cloud data formats.
        
        Args:
            dataset_type: Dataset type string
            
        Returns:
            True if dataset requires 3D visualization
        """
        # Import here to avoid circular imports
        from data.viewer.backend.backend import DATASET_FORMATS
        
        # Check if the dataset type has point_cloud in its input format
        dataset_format = DATASET_FORMATS.get(dataset_type, {})
        input_format = dataset_format.get('input_format', {})
        return 'point_cloud' in input_format
