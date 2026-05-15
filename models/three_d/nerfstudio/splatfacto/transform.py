"""Transform Nerfstudio Splatfacto Gaussian parameters in the dataset world frame.

The public transform path preserves the caller's floating precision at the
module boundary. `float64` inputs stay `float64` throughout. `float32` inputs
stay `float32` except for the internal SH-rotation solve, which temporarily
uses `float64` and converts the solved SH rotation matrix back to the caller's
target dtype before applying it.
"""

import math
from numbers import Real
from typing import Optional

import torch
from gsplat.cuda._torch_impl import _eval_sh_bases_fast
from nerfstudio.pipelines.base_pipeline import Pipeline

from data.structures.three_d.camera.rotation.quaternion import rotmat_to_quat

NUM_REST_COEFFICIENTS_TO_SH_DEGREE = {
    0: 0,
    3: 1,
    8: 2,
    15: 3,
    24: 4,
}
SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


def apply_transform_to_splatfacto(
    pipeline: Pipeline,
    rotation: Optional[torch.Tensor] = None,
    translation: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> None:
    """Apply a world-frame similarity transform to a loaded Splatfacto pipeline in-place.

    Args:
        pipeline: Nerfstudio pipeline whose model exposes `means`, `quats`, `scales`, and `features_rest`.
        rotation: Optional `float [3, 3]` proper world rotation matrix.
        translation: Optional `float [3]` world translation vector.
        scale: Optional positive global scale factor.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(pipeline, Pipeline), (
            "`pipeline` must be a Nerfstudio `Pipeline`. " f"Got {type(pipeline)=}."
        )
        assert hasattr(pipeline, "model"), (
            "`pipeline` must expose a `model` attribute. " f"Got {type(pipeline)=}."
        )
        for attribute_name in ["means", "quats", "scales", "features_rest"]:
            assert hasattr(pipeline.model, attribute_name), (
                "`pipeline.model` must expose all required Splatfacto tensor "
                "attributes. "
                f"Got missing {attribute_name=} and {type(pipeline.model)=}."
            )
        _validate_model_attributes(
            means=pipeline.model.means,
            quats=pipeline.model.quats,
            scales=pipeline.model.scales,
            features_rest=pipeline.model.features_rest,
        )
        _validate_transform_arguments(
            rotation=rotation,
            translation=translation,
            scale=scale,
        )
        if rotation is not None:
            assert rotation.device == pipeline.model.means.device, (
                "`rotation` must live on the same device as the Splatfacto model "
                "tensors. "
                f"Got {rotation.device=} and {pipeline.model.means.device=}."
            )
            assert rotation.dtype == pipeline.model.means.dtype, (
                "`rotation` must use the same dtype as the Splatfacto model "
                "tensors. "
                f"Got {rotation.dtype=} and {pipeline.model.means.dtype=}."
            )
        if translation is not None:
            assert translation.device == pipeline.model.means.device, (
                "`translation` must live on the same device as the Splatfacto "
                "model tensors. "
                f"Got {translation.device=} and {pipeline.model.means.device=}."
            )
            assert translation.dtype == pipeline.model.means.dtype, (
                "`translation` must use the same dtype as the Splatfacto model "
                "tensors. "
                f"Got {translation.dtype=} and {pipeline.model.means.dtype=}."
            )

    _validate_inputs()

    with torch.no_grad():
        _apply_scale_transform(
            means=pipeline.model.means,
            scales=pipeline.model.scales,
            scale=scale,
        )
        _apply_rotation_transform(
            means=pipeline.model.means,
            quats=pipeline.model.quats,
            features_rest=pipeline.model.features_rest,
            rotation=rotation,
        )
        _apply_translation_transform(
            means=pipeline.model.means,
            translation=translation,
        )


def _validate_model_attributes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    features_rest: torch.Tensor,
) -> None:
    """Validate Splatfacto Gaussian parameter tensors.

    Args:
        means: `float [N, 3]` Gaussian centers in world coordinates.
        quats: `float [N, 4]` scalar-first Gaussian quaternions in `wxyz`.
        scales: `float [N, 3]` Gaussian log-scales along local principal axes.
        features_rest: `float [N, K, 3]` non-DC SH coefficients in gsplat basis.

    Returns:
        None.
    """
    assert isinstance(means, torch.Tensor), (
        "`means` must be a `torch.Tensor`. " f"Got {type(means)=}."
    )
    assert means.ndim == 2 and means.shape[1] == 3, (
        "`means` must have shape `[N, 3]`. " f"Got {means.shape=}."
    )
    assert torch.is_floating_point(means), (
        "`means` must use a floating dtype. " f"Got {means.dtype=}."
    )
    assert means.dtype in SUPPORTED_FLOAT_DTYPES, (
        "`means.dtype` must be one of the supported floating dtypes. "
        f"Got {means.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
    )
    assert torch.isfinite(means).all(), (
        "All values in `means` must be finite. "
        f"Got {means.shape=} and {means.dtype=}."
    )

    assert isinstance(quats, torch.Tensor), (
        "`quats` must be a `torch.Tensor`. " f"Got {type(quats)=}."
    )
    assert quats.ndim == 2 and quats.shape == (means.shape[0], 4), (
        "`quats` must have shape `[N, 4]` aligned with `means`. "
        f"Got {quats.shape=} and {means.shape=}."
    )
    assert torch.is_floating_point(quats), (
        "`quats` must use a floating dtype. " f"Got {quats.dtype=}."
    )
    assert quats.dtype in SUPPORTED_FLOAT_DTYPES, (
        "`quats.dtype` must be one of the supported floating dtypes. "
        f"Got {quats.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
    )
    assert torch.isfinite(quats).all(), (
        "All values in `quats` must be finite. "
        f"Got {quats.shape=} and {quats.dtype=}."
    )
    assert quats.device == means.device, (
        "`quats` and `means` must live on the same device. "
        f"Got {quats.device=} and {means.device=}."
    )
    assert quats.dtype == means.dtype, (
        "`quats` and `means` must use the same dtype. "
        f"Got {quats.dtype=} and {means.dtype=}."
    )
    assert torch.all(torch.linalg.norm(quats, dim=1) > 0.0), (
        "Every quaternion in `quats` must have strictly positive norm. "
        f"Got min_quat_norm={torch.linalg.norm(quats, dim=1).min().item()} "
        f"with {quats.shape=}."
    )

    assert isinstance(scales, torch.Tensor), (
        "`scales` must be a `torch.Tensor`. " f"Got {type(scales)=}."
    )
    assert scales.ndim == 2 and scales.shape == means.shape, (
        "`scales` must have shape `[N, 3]` aligned with `means`. "
        f"Got {scales.shape=} and {means.shape=}."
    )
    assert torch.is_floating_point(scales), (
        "`scales` must use a floating dtype. " f"Got {scales.dtype=}."
    )
    assert scales.dtype in SUPPORTED_FLOAT_DTYPES, (
        "`scales.dtype` must be one of the supported floating dtypes. "
        f"Got {scales.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
    )
    assert torch.isfinite(scales).all(), (
        "All values in `scales` must be finite. "
        f"Got {scales.shape=} and {scales.dtype=}."
    )
    assert scales.device == means.device, (
        "`scales` and `means` must live on the same device. "
        f"Got {scales.device=} and {means.device=}."
    )
    assert scales.dtype == means.dtype, (
        "`scales` and `means` must use the same dtype. "
        f"Got {scales.dtype=} and {means.dtype=}."
    )

    assert isinstance(features_rest, torch.Tensor), (
        "`features_rest` must be a `torch.Tensor`. " f"Got {type(features_rest)=}."
    )
    assert features_rest.ndim == 3, (
        "`features_rest` must have shape `[N, K, 3]`. " f"Got {features_rest.shape=}."
    )
    assert features_rest.shape[0] == means.shape[0] and features_rest.shape[2] == 3, (
        "`features_rest` must align with `means` in batch size and must have "
        f"exactly 3 RGB channels. Got {features_rest.shape=} and {means.shape=}."
    )
    assert features_rest.shape[1] in NUM_REST_COEFFICIENTS_TO_SH_DEGREE, (
        "`features_rest.shape[1]` must be a supported non-DC SH coefficient count "
        f"for gsplat rotation. Got {features_rest.shape=} and "
        f"supported_counts={tuple(NUM_REST_COEFFICIENTS_TO_SH_DEGREE)}."
    )
    assert torch.is_floating_point(features_rest), (
        "`features_rest` must use a floating dtype. " f"Got {features_rest.dtype=}."
    )
    assert features_rest.dtype in SUPPORTED_FLOAT_DTYPES, (
        "`features_rest.dtype` must be one of the supported floating dtypes. "
        f"Got {features_rest.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
    )
    assert torch.isfinite(features_rest).all(), (
        "All values in `features_rest` must be finite. "
        f"Got {features_rest.shape=} and {features_rest.dtype=}."
    )
    assert features_rest.device == means.device, (
        "`features_rest` and `means` must live on the same device. "
        f"Got {features_rest.device=} and {means.device=}."
    )
    assert features_rest.dtype == means.dtype, (
        "`features_rest` and `means` must use the same dtype. "
        f"Got {features_rest.dtype=} and {means.dtype=}."
    )


def _validate_transform_arguments(
    rotation: Optional[torch.Tensor],
    translation: Optional[torch.Tensor],
    scale: Optional[float],
) -> None:
    """Validate rigid-plus-scale world transform inputs.

    Args:
        rotation: Optional `float [3, 3]` proper rotation matrix in column-vector convention.
        translation: Optional `float [3]` world translation vector.
        scale: Optional positive global scale factor.

    Returns:
        None.
    """
    if rotation is not None:
        assert isinstance(rotation, torch.Tensor), (
            "`rotation` must be a `torch.Tensor` when provided. "
            f"Got {type(rotation)=}."
        )
        assert rotation.ndim == 2 and rotation.shape == (3, 3), (
            "`rotation` must have shape `[3, 3]`. " f"Got {rotation.shape=}."
        )
        assert torch.is_floating_point(rotation), (
            "`rotation` must use a floating dtype. " f"Got {rotation.dtype=}."
        )
        assert rotation.dtype in SUPPORTED_FLOAT_DTYPES, (
            "`rotation.dtype` must be one of the supported floating dtypes. "
            f"Got {rotation.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
        )
        assert torch.isfinite(rotation).all(), (
            "All values in `rotation` must be finite. "
            f"Got {rotation.shape=} and {rotation.dtype=}."
        )
        assert torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            torch.eye(
                3,
                dtype=rotation.dtype,
                device=rotation.device,
            ),
            atol=1.0e-5,
            rtol=1.0e-5,
        ), (
            "`rotation` must be orthonormal. Got max_orthogonality_error="
            f"{torch.max(torch.abs(rotation.transpose(0, 1) @ rotation - torch.eye(3, dtype=rotation.dtype, device=rotation.device))).item()} "
            f"det_rotation={torch.linalg.det(rotation).item()} and {rotation=}."
        )
        assert abs(torch.linalg.det(rotation).item() - 1.0) <= 1.0e-5, (
            "`rotation` must be a proper rotation with determinant `+1`. "
            f"Got det_rotation={torch.linalg.det(rotation).item()} and "
            f"{rotation=}."
        )

    if translation is not None:
        assert isinstance(translation, torch.Tensor), (
            "`translation` must be a `torch.Tensor` when provided. "
            f"Got {type(translation)=}."
        )
        assert translation.ndim == 1 and translation.shape == (3,), (
            "`translation` must have shape `[3]`. " f"Got {translation.shape=}."
        )
        assert torch.is_floating_point(translation), (
            "`translation` must use a floating dtype. " f"Got {translation.dtype=}."
        )
        assert translation.dtype in SUPPORTED_FLOAT_DTYPES, (
            "`translation.dtype` must be one of the supported floating dtypes. "
            f"Got {translation.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
        )
        assert torch.isfinite(translation).all(), (
            "All values in `translation` must be finite. "
            f"Got {translation.shape=} and {translation.dtype=}."
        )

    if scale is not None:
        assert isinstance(scale, Real) and not isinstance(scale, bool), (
            "`scale` must be a real number and must not be a bool when provided. "
            f"Got {type(scale)=}."
        )
        assert math.isfinite(float(scale)), (
            "`scale` must be finite. " f"Got scale_value={float(scale)}."
        )
        assert float(scale) > 0.0, (
            "`scale` must be strictly positive. " f"Got scale_value={float(scale)}."
        )

    if rotation is not None and translation is not None:
        assert translation.device == rotation.device, (
            "`rotation` and `translation` must live on the same device. "
            f"Got {rotation.device=} and {translation.device=}."
        )
        assert translation.dtype == rotation.dtype, (
            "`rotation` and `translation` must use the same dtype. "
            f"Got {rotation.dtype=} and {translation.dtype=}."
        )


def _apply_scale_transform(
    means: torch.Tensor,
    scales: torch.Tensor,
    scale: Optional[float],
) -> None:
    """Apply the scale phase of a world-frame similarity transform in-place.

    Args:
        means: `float [N, 3]` Gaussian centers in world coordinates.
        scales: `float [N, 3]` Gaussian log-scales along local principal axes.
        scale: Optional positive global scale factor.

    Returns:
        None.
    """
    if scale is None:
        return

    scale_value = float(scale)
    means.mul_(scale_value)
    scales.add_(math.log(scale_value))


def _apply_rotation_transform(
    means: torch.Tensor,
    quats: torch.Tensor,
    features_rest: torch.Tensor,
    rotation: Optional[torch.Tensor],
) -> None:
    """Apply the rotation phase of a world-frame similarity transform in-place.

    Args:
        means: `float [N, 3]` Gaussian centers in world coordinates.
        quats: `float [N, 4]` scalar-first Gaussian quaternions in `wxyz`.
        features_rest: `float [N, K, 3]` non-DC SH coefficients in gsplat basis.
        rotation: Optional `float [3, 3]` proper world rotation matrix.

    Returns:
        None.
    """
    if rotation is None:
        return

    means[:] = means @ rotation.transpose(0, 1)
    rotation_quaternion = _build_rotation_quaternion(rotation=rotation)
    quats[:] = _left_multiply_quaternions(
        left_quaternion=rotation_quaternion,
        right_quaternions=quats,
    )
    features_rest[:] = _rotate_features_rest(
        features_rest=features_rest,
        rotation=rotation,
    )


def _apply_translation_transform(
    means: torch.Tensor,
    translation: Optional[torch.Tensor],
) -> None:
    """Apply the translation phase of a world-frame similarity transform in-place.

    Args:
        means: `float [N, 3]` Gaussian centers in world coordinates.
        translation: Optional `float [3]` world translation vector.

    Returns:
        None.
    """
    if translation is None:
        return

    means.add_(translation.view(1, 3))


def _build_rotation_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a world rotation matrix into a scalar-first quaternion.

    Args:
        rotation: `float [3, 3]` proper world rotation matrix.

    Returns:
        `float [4]` scalar-first quaternion in `wxyz`.
    """
    return rotmat_to_quat(rotation.unsqueeze(0))[0]


def _left_multiply_quaternions(
    left_quaternion: torch.Tensor,
    right_quaternions: torch.Tensor,
) -> torch.Tensor:
    """Left-multiply scalar-first quaternions in `wxyz` convention.

    Args:
        left_quaternion: `float [4]` left quaternion in `wxyz`.
        right_quaternions: `float [N, 4]` right quaternions in `wxyz`.

    Returns:
        `float [N, 4]` Hamilton products `left_quaternion * right_quaternions`.
    """
    left_w, left_x, left_y, left_z = left_quaternion.unbind(dim=0)
    right_w, right_x, right_y, right_z = right_quaternions.unbind(dim=1)

    return torch.stack(
        [
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ],
        dim=1,
    )


def _rotate_features_rest(
    features_rest: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Rotate non-DC SH coefficients under a world-frame rotation.

    Args:
        features_rest: `float [N, K, 3]` non-DC SH coefficients in gsplat basis.
        rotation: `float [3, 3]` proper world rotation matrix.

    Returns:
        `float [N, K, 3]` rotated non-DC SH coefficients.
    """
    num_rest_coefficients = features_rest.shape[1]
    if num_rest_coefficients == 0:
        return features_rest.clone()

    rest_sh_rotation_matrix = _build_rest_sh_rotation_matrix(
        rotation=rotation,
        num_rest_coefficients=num_rest_coefficients,
        dtype=features_rest.dtype,
        device=features_rest.device,
    )
    return torch.einsum(
        "ij,njc->nic",
        rest_sh_rotation_matrix,
        features_rest,
    )


def _build_rest_sh_rotation_matrix(
    rotation: torch.Tensor,
    num_rest_coefficients: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build the gsplat SH coefficient rotation matrix for non-DC coefficients.

    Args:
        rotation: `float [3, 3]` proper world rotation matrix.
        num_rest_coefficients: Number of non-DC SH coefficients per color channel.
        dtype: Target dtype of the returned coefficient rotation matrix.
        device: Target device of the returned coefficient rotation matrix.

    Returns:
        `float [K, K]` SH coefficient rotation matrix for `features_rest`.
    """

    def _validate_inputs() -> None:
        assert isinstance(num_rest_coefficients, int) and not isinstance(
            num_rest_coefficients,
            bool,
        ), (
            "`num_rest_coefficients` must be an integer and must not be a bool. "
            f"Got {type(num_rest_coefficients)=}."
        )
        assert num_rest_coefficients in NUM_REST_COEFFICIENTS_TO_SH_DEGREE, (
            "`num_rest_coefficients` must be one of the supported non-DC SH "
            f"coefficient counts. Got {num_rest_coefficients=} and "
            f"supported_counts={tuple(NUM_REST_COEFFICIENTS_TO_SH_DEGREE)}."
        )
        assert isinstance(dtype, torch.dtype), (
            "`dtype` must be a `torch.dtype`. " f"Got {type(dtype)=}."
        )
        assert dtype in SUPPORTED_FLOAT_DTYPES, (
            "The SH rotation target `dtype` must be supported. "
            f"Got {dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
        )
        assert isinstance(device, torch.device), (
            "`device` must be a `torch.device`. " f"Got {type(device)=}."
        )

    _validate_inputs()

    if num_rest_coefficients == 0:
        return torch.empty(
            (0, 0),
            dtype=dtype,
            device=device,
        )

    full_basis_dim = num_rest_coefficients + 1
    num_sphere_directions = max(128, 8 * full_basis_dim)
    sphere_directions = _build_fibonacci_sphere_directions(
        num_directions=num_sphere_directions,
        dtype=torch.float64,
        device=rotation.device,
    )
    assert rotation.dtype in SUPPORTED_FLOAT_DTYPES, (
        "`rotation.dtype` must be supported before converting `rotation` to "
        f"`torch.float64`. Got {rotation.dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
    )
    rotated_sphere_directions = sphere_directions @ rotation.to(dtype=torch.float64)
    full_basis = _eval_sh_bases_fast(full_basis_dim, sphere_directions)
    rotated_full_basis = _eval_sh_bases_fast(full_basis_dim, rotated_sphere_directions)
    full_sh_rotation_matrix = torch.linalg.lstsq(
        full_basis,
        rotated_full_basis,
    ).solution
    rest_sh_rotation_matrix = full_sh_rotation_matrix[1:, 1:]
    assert rest_sh_rotation_matrix.dtype == torch.float64, (
        "The solved non-DC SH rotation matrix must be `torch.float64` before "
        f"converting it to the target dtype. Got {rest_sh_rotation_matrix.dtype=}."
    )
    return rest_sh_rotation_matrix.to(
        dtype=dtype,
        device=device,
    )


def _build_fibonacci_sphere_directions(
    num_directions: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic unit directions on the sphere.

    Args:
        num_directions: Number of unit directions to generate.
        dtype: Floating dtype of the generated directions.
        device: Device of the generated directions.

    Returns:
        `float [M, 3]` row-vector unit directions.
    """

    def _validate_inputs() -> None:
        assert isinstance(num_directions, int) and not isinstance(
            num_directions, bool
        ), (
            "`num_directions` must be an integer and must not be a bool. "
            f"Got {type(num_directions)=}."
        )
        assert num_directions > 0, (
            "`num_directions` must be strictly positive. " f"Got {num_directions=}."
        )
        assert isinstance(dtype, torch.dtype), (
            "`dtype` must be a `torch.dtype`. " f"Got {type(dtype)=}."
        )
        assert dtype in SUPPORTED_FLOAT_DTYPES, (
            "The sphere-direction `dtype` must be supported. "
            f"Got {dtype=} and {SUPPORTED_FLOAT_DTYPES=}."
        )
        assert isinstance(device, torch.device), (
            "`device` must be a `torch.device`. " f"Got {type(device)=}."
        )

    _validate_inputs()

    point_indices = torch.arange(
        num_directions,
        dtype=dtype,
        device=device,
    )
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y_values = 1.0 - (2.0 * point_indices + 1.0) / num_directions
    radial_values = torch.sqrt(torch.clamp(1.0 - y_values.square(), min=0.0))
    azimuth_values = golden_angle * point_indices
    x_values = torch.cos(azimuth_values) * radial_values
    z_values = torch.sin(azimuth_values) * radial_values
    return torch.stack([x_values, y_values, z_values], dim=1)
