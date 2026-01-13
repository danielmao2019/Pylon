"""Factory for creating Point Cloud Level-of-Detail (LOD) functions.

This factory creates callable functions for applying LOD to point clouds
based on the specified LOD type and configuration parameters.
"""
from typing import Dict, Any, Callable, Optional
from functools import partial

from data.transforms.vision_3d.pclod.continuous_lod import ContinuousLOD
from data.transforms.vision_3d.pclod.discrete_lod import DiscreteLOD
from data.transforms.vision_3d.pclod.density_lod import DensityLOD
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def create_lod_function(
    lod_type: str = "none",
    lod_config: Optional[Dict[str, Any]] = None,
    point_cloud_id: Optional[str] = None
) -> Callable[[PointCloud], PointCloud]:
    """Create a LOD function based on the specified type and configuration.

    Args:
        lod_type: Type of LOD ("none", "density", "continuous", or "discrete")
        lod_config: LOD configuration dictionary:
            - For "none": Should be None or empty dict
            - For "density": {"density": int} where density is percentage (1-100)
            - For "continuous": {"camera_state": dict, ...other params...}
            - For "discrete": {"camera_state": dict, ...other params...}
        point_cloud_id: Unique identifier for LOD caching (required for "density" and "discrete")

    Returns:
        A callable function that takes a PointCloud and returns a PointCloud

    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(lod_type, str), f"lod_type must be str, got {type(lod_type)}"
    assert lod_type in ["none", "density", "continuous", "discrete"], \
        f"lod_type must be one of ['none', 'density', 'continuous', 'discrete'], got {lod_type}"

    # Validate lod_config based on lod_type
    if lod_type == "none":
        # For "none", lod_config should be None or empty
        assert lod_config is None or len(lod_config) == 0, \
            f"lod_type='none' should have empty or None lod_config, got {list(lod_config.keys()) if lod_config else None}"

        # Return identity function (no processing)
        def _identity(pc: PointCloud) -> PointCloud:
            assert isinstance(pc, PointCloud), f"{type(pc)=}"
            return pc

        return _identity

    elif lod_type == "density":
        # For "density", lod_config must have exactly 'density' key
        assert lod_config is not None, "lod_config is required for lod_type='density'"
        assert set(lod_config.keys()) == {"density"}, \
            f"lod_type='density' must have only 'density' key in lod_config, got keys: {list(lod_config.keys())}"
        assert isinstance(lod_config["density"], int), \
            f"density must be int, got {type(lod_config['density'])}"
        assert 1 <= lod_config["density"] <= 100, \
            f"density must be 1-100, got {lod_config['density']}"
        assert point_cloud_id is not None, "point_cloud_id is required for lod_type='density'"

        # Create density LOD function
        density_percentage = lod_config["density"]
        if density_percentage == 100:
            # No subsampling needed
            return lambda pc: pc

        density_lod = DensityLOD()
        return partial(
            density_lod.subsample,
            density_percentage=density_percentage,
            point_cloud_id=point_cloud_id
        )

    elif lod_type == "continuous":
        # For "continuous", must have 'camera_state' and no 'density'
        assert lod_config is not None, f"lod_config is required for lod_type='continuous'"
        assert "camera_state" in lod_config, \
            f"lod_type='continuous' requires 'camera_state' in lod_config, got keys: {list(lod_config.keys())}"
        assert "density" not in lod_config, \
            f"lod_type='continuous' should not have 'density' in lod_config, got keys: {list(lod_config.keys())}"
        assert isinstance(lod_config["camera_state"], dict), \
            f"camera_state must be dict, got {type(lod_config['camera_state'])}"

        # Extract camera_state and constructor params
        camera_state = lod_config["camera_state"]
        constructor_params = {k: v for k, v in lod_config.items() if k != "camera_state"}

        # Create continuous LOD function
        continuous_lod = ContinuousLOD(**constructor_params)
        return partial(
            continuous_lod.subsample,
            camera_state=camera_state
        )

    elif lod_type == "discrete":
        # For "discrete", must have 'camera_state' and no 'density'
        assert lod_config is not None, f"lod_config is required for lod_type='discrete'"
        assert "camera_state" in lod_config, \
            f"lod_type='discrete' requires 'camera_state' in lod_config, got keys: {list(lod_config.keys())}"
        assert "density" not in lod_config, \
            f"lod_type='discrete' should not have 'density' in lod_config, got keys: {list(lod_config.keys())}"
        assert isinstance(lod_config["camera_state"], dict), \
            f"camera_state must be dict, got {type(lod_config['camera_state'])}"
        assert point_cloud_id is not None, "point_cloud_id is required for lod_type='discrete'"

        # Extract camera_state and constructor params
        camera_state = lod_config["camera_state"]
        constructor_params = {k: v for k, v in lod_config.items() if k != "camera_state"}

        # Create discrete LOD function
        discrete_lod = DiscreteLOD(**constructor_params)
        return partial(
            discrete_lod.subsample,
            camera_state=camera_state,
            point_cloud_id=point_cloud_id
        )

    else:
        # This should never happen due to earlier validation
        raise ValueError(f"Unknown LOD type: {lod_type}")
