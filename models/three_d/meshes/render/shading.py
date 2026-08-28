"""
Spherical-harmonic shading of surface normals.
"""

import math

import torch


def compute_sh_shading(
    normals: torch.Tensor,
    sh_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate spherical-harmonic shading over surface normals at whatever band
    count `sh_coefficients` carries.

    The basis is the real spherical-harmonic basis in band order: ascending
    degree `l`, and within a degree ascending order `m`, each term carrying the
    Condon-Shortley sign. The coefficients are consumed exactly as given and
    nothing is added to them, so a zero coefficient vector shades black; a
    caller whose model parameterizes its lighting as an offset from an ambient
    baseline adds that baseline to its own coefficients before calling.

    Args:
        normals: Unit-length surface normals in the frame the lighting is
            defined in, shape `[..., N, 3]` float32, xyz-ordered. The leading
            batch dims must equal `sh_coefficients`' leading batch dims, and
            both tensors must live on the same device.
        sh_coefficients: Spherical-harmonic lighting coefficients, shape
            `[..., B * 3]` float32, laid out channel-major as three contiguous
            `B`-band blocks in R, G, B order. `B` must be a perfect square, the
            square of the spherical-harmonic order it implies.

    Returns:
        The per-normal RGB shading factor the caller multiplies its albedo by,
        shape `[..., N, 3]` float32 on `normals.device`, UNCLAMPED (it may
        exceed `[0, 1]` or go negative).
    """

    def _validate_inputs() -> None:
        assert isinstance(normals, torch.Tensor), (
            "Expected normals to be a torch.Tensor. " f"{type(normals)=}"
        )
        assert normals.ndim >= 2, (
            "Expected normals to have shape [..., N, 3], so at least 2 dims. "
            f"{normals.shape=}"
        )
        assert normals.shape[-1] == 3, (
            "Expected normals to have shape [..., N, 3]. " f"{normals.shape=}"
        )
        assert normals.dtype == torch.float32, (
            "Expected normals to be float32. " f"{normals.dtype=}"
        )
        assert isinstance(sh_coefficients, torch.Tensor), (
            "Expected sh_coefficients to be a torch.Tensor. "
            f"{type(sh_coefficients)=}"
        )
        assert sh_coefficients.ndim >= 1, (
            "Expected sh_coefficients to have shape [..., B * 3], so at least 1 dim. "
            f"{sh_coefficients.shape=}"
        )
        assert sh_coefficients.dtype == torch.float32, (
            "Expected sh_coefficients to be float32. " f"{sh_coefficients.dtype=}"
        )
        assert sh_coefficients.shape[-1] % 3 == 0, (
            "Expected sh_coefficients to carry three equal-length RGB band blocks, "
            "so its last dim must be divisible by 3. "
            f"{sh_coefficients.shape=}"
        )
        assert sh_coefficients.shape[-1] // 3 > 0, (
            "Expected sh_coefficients to carry at least one spherical-harmonic "
            "band. "
            f"{sh_coefficients.shape=}"
        )
        assert (
            math.isqrt(sh_coefficients.shape[-1] // 3) ** 2
            == sh_coefficients.shape[-1] // 3
        ), (
            "Expected sh_coefficients' band count to be a perfect square, naming "
            "the spherical-harmonic order whose bands it carries. "
            f"{sh_coefficients.shape=}"
        )
        assert sh_coefficients.shape[:-1] == normals.shape[:-2], (
            "Expected sh_coefficients' batch dims to equal normals' batch dims. "
            f"{sh_coefficients.shape=}, {normals.shape=}"
        )
        assert sh_coefficients.device == normals.device, (
            "Expected sh_coefficients and normals to be on the same device. "
            f"{sh_coefficients.device=}, {normals.device=}"
        )

    _validate_inputs()

    num_bands = sh_coefficients.shape[-1] // 3
    order = math.isqrt(num_bands)

    # === Spherical-harmonic basis over the normals ===
    pi = math.pi
    sqrt_two = math.sqrt(2.0)
    x = normals[..., 0]
    y = normals[..., 1]
    z = normals[..., 2]
    radius_xy = torch.sqrt(torch.clamp(1.0 - z**2, min=0.0))
    phi = torch.atan2(y, x)

    associated_legendre = {}
    for degree_order in range(order):
        if degree_order == 0:
            p_mm = torch.ones_like(z)
        else:
            p_mm = (
                -(2 * degree_order - 1)
                * radius_xy
                * associated_legendre[(degree_order - 1, degree_order - 1)]
            )
        associated_legendre[(degree_order, degree_order)] = p_mm

        if degree_order + 1 < order:
            associated_legendre[(degree_order + 1, degree_order)] = (
                (2 * degree_order + 1) * z * p_mm
            )

        for degree in range(degree_order + 2, order):
            associated_legendre[(degree, degree_order)] = (
                (2 * degree - 1) * z * associated_legendre[(degree - 1, degree_order)]
                - (degree + degree_order - 1)
                * associated_legendre[(degree - 2, degree_order)]
            ) / float(degree - degree_order)

    basis_terms = []
    for degree in range(order):
        for signed_order in range(-degree, degree + 1):
            abs_order = abs(signed_order)
            factorial_ratio = 1.0
            for factor in range(degree - abs_order + 1, degree + abs_order + 1):
                factorial_ratio /= float(factor)
            normalizer = ((2 * degree + 1) * factorial_ratio / (4.0 * pi)) ** 0.5
            legendre = associated_legendre[(degree, abs_order)]

            if signed_order < 0:
                term = sqrt_two * normalizer * legendre * torch.sin(abs_order * phi)
            elif signed_order == 0:
                term = normalizer * legendre
            else:
                term = sqrt_two * normalizer * legendre * torch.cos(abs_order * phi)
            basis_terms.append(term.unsqueeze(dim=-1))

    basis = torch.cat(basis_terms, dim=-1)

    # === Contraction of the basis against the coefficients over the bands ===
    # The channel-major coefficient blocks are unflattened to [..., 3, B], then transposed to put the bands on the contraction axis as [..., B, 3].
    coefficients = sh_coefficients.reshape(
        *sh_coefficients.shape[:-1], 3, num_bands
    ).transpose(-1, -2)
    # One matmul per channel, never a single [..., N, B] @ [..., B, 3]: on CUDA the fused form can pick a different reduction and shift results by small numeric roundoff relative to this channel-wise contraction.
    shading = torch.cat(
        [
            basis @ coefficients[..., :1],
            basis @ coefficients[..., 1:2],
            basis @ coefficients[..., 2:],
        ],
        dim=-1,
    )
    return shading
