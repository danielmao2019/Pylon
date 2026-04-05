"""Shared weight-config helpers for texture extraction."""

from typing import Any, Dict

WEIGHTS_CFG_ALLOWED_KEYS = [
    "weights",
    "normals_weight_power",
    "normals_weight_threshold",
    "multi_view_robustness",
    "robustness_tau",
    "first_frame_blending_weight_power",
]


def validate_weights_cfg(
    weights_cfg: Dict[str, Any],
) -> None:
    """Validate one texture-extraction weights config.

    Args:
        weights_cfg: Weights configuration dictionary.

    Returns:
        None.
    """
    assert isinstance(weights_cfg, dict), (
        "Expected `weights_cfg` to be a `dict`. " f"{type(weights_cfg)=}"
    )
    assert set(weights_cfg.keys()).issubset(set(WEIGHTS_CFG_ALLOWED_KEYS)), (
        "Expected `weights_cfg` keys to stay within the allowed set. "
        f"{weights_cfg.keys()=} {WEIGHTS_CFG_ALLOWED_KEYS=}"
    )
    assert "weights" not in weights_cfg or (
        isinstance(weights_cfg["weights"], str)
        and weights_cfg["weights"] in ("visible", "normals")
    ), f"{weights_cfg=}"
    assert "normals_weight_power" not in weights_cfg or (
        isinstance(weights_cfg["normals_weight_power"], float)
        and weights_cfg["normals_weight_power"] > 0.0
    ), f"{weights_cfg=}"
    assert "normals_weight_threshold" not in weights_cfg or (
        isinstance(weights_cfg["normals_weight_threshold"], float)
        and weights_cfg["normals_weight_threshold"] >= 0.0
        and weights_cfg["normals_weight_threshold"] <= 1.0
    ), f"{weights_cfg=}"
    assert "multi_view_robustness" not in weights_cfg or (
        isinstance(weights_cfg["multi_view_robustness"], str)
        and weights_cfg["multi_view_robustness"] in ("none", "residual_gaussian")
    ), f"{weights_cfg=}"
    assert "robustness_tau" not in weights_cfg or (
        isinstance(weights_cfg["robustness_tau"], float)
        and weights_cfg["robustness_tau"] > 0.0
    ), f"{weights_cfg=}"
    assert "first_frame_blending_weight_power" not in weights_cfg or (
        isinstance(weights_cfg["first_frame_blending_weight_power"], float)
        and weights_cfg["first_frame_blending_weight_power"] > 0.0
    ), f"{weights_cfg=}"


def normalize_weights_cfg(
    weights_cfg: Dict[str, Any],
    default_weights: str,
) -> Dict[str, Any]:
    """Normalize one texture-extraction weights config.

    Args:
        weights_cfg: Weights configuration dictionary.
        default_weights: Default weighting mode when `weights` is omitted.

    Returns:
        Normalized weights configuration dictionary.
    """
    weights_cfg = weights_cfg.copy()
    if "weights" not in weights_cfg:
        weights_cfg["weights"] = default_weights
    if "normals_weight_power" not in weights_cfg:
        weights_cfg["normals_weight_power"] = 1.0
    if "normals_weight_threshold" not in weights_cfg:
        weights_cfg["normals_weight_threshold"] = 0.0
    if "multi_view_robustness" not in weights_cfg:
        weights_cfg["multi_view_robustness"] = "none"
    if "robustness_tau" not in weights_cfg:
        weights_cfg["robustness_tau"] = 0.2
    if "first_frame_blending_weight_power" not in weights_cfg:
        weights_cfg["first_frame_blending_weight_power"] = 2.0
    return weights_cfg
