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
    Evaluate spherical-harmonic shading over surface normals, at whatever band
    count `sh_coefficients` carries.

    The basis is the real spherical-harmonic basis in Deep3DFaceRecon band
    order: ascending degree `l`, and within a degree ascending order `m`, each
    term carrying the `(-1)^m` sign of that convention and scaled by the
    Deep3DFaceRecon per-degree Lambertian factor `a[l]` and normalization
    `c[l]`. Those constants are defined for degrees `0..2` only, so the band
    count may name orders 1, 2, or 3 (1, 4, or 9 bands). The coefficients are
    consumed exactly as given and nothing is added to them, so a zero
    coefficient vector shades black; a caller whose model parameterizes its
    lighting as an offset from an ambient baseline adds that baseline to its own
    coefficients before calling.

    Args:
        normals: Unit-length surface normals in the frame the lighting is
            defined in, shape `[..., N, 3]` float32, xyz-ordered. The leading
            batch dims must equal `sh_coefficients`' leading batch dims, and
            both tensors must live on the same device.
        sh_coefficients: Spherical-harmonic lighting coefficients, shape
            `[..., B * 3]` float32, laid out channel-major as three contiguous
            `B`-band blocks in R, G, B order (the `BFM09Params.gamma` layout).
            `B` must be a perfect square, the square of the spherical-harmonic
            order it implies.

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
        num_bands = sh_coefficients.shape[-1] // 3
        order = round(math.sqrt(num_bands))
        assert order * order == num_bands, (
            "Expected sh_coefficients' band count to be a perfect square, naming "
            "the spherical-harmonic order whose bands it carries. "
            f"{num_bands=}, {order=}, {sh_coefficients.shape=}"
        )
        assert 1 <= order <= 3, (
            "Expected a spherical-harmonic order the Deep3DFaceRecon constant set "
            "defines, i.e. at most order 3 (9 bands). "
            f"{order=}, {num_bands=}, {sh_coefficients.shape=}"
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

    # === Spherical-harmonic basis over the normals ===
    # Deep3DFaceRecon's per-degree Lambertian factor `a` and real-SH normalization `c`, indexed by degree; the basis terms below are band-ordered and sliced to the band count the coefficients carry.
    sh_a = [math.pi, 2 * math.pi / math.sqrt(3.0), 2 * math.pi / math.sqrt(8.0)]
    sh_c = [
        1 / math.sqrt(4 * math.pi),
        math.sqrt(3.0) / math.sqrt(4 * math.pi),
        3 * math.sqrt(5.0) / math.sqrt(12 * math.pi),
    ]
    x = normals[..., :1]
    y = normals[..., 1:2]
    z = normals[..., 2:]
    basis = torch.cat(
        [
            sh_a[0] * sh_c[0] * torch.ones_like(x),
            -sh_a[1] * sh_c[1] * y,
            sh_a[1] * sh_c[1] * z,
            -sh_a[1] * sh_c[1] * x,
            sh_a[2] * sh_c[2] * x * y,
            -sh_a[2] * sh_c[2] * y * z,
            0.5 * sh_a[2] * sh_c[2] / math.sqrt(3.0) * (3 * z**2 - 1),
            -sh_a[2] * sh_c[2] * x * z,
            0.5 * sh_a[2] * sh_c[2] * (x**2 - y**2),
        ][:num_bands],
        dim=-1,
    )

    # === Contraction of the basis against the coefficients over the bands ===
    # The channel-major coefficient blocks are unflattened to [..., 3, B], then transposed to put the bands on the contraction axis as [..., B, 3].
    coefficients = sh_coefficients.reshape(
        *sh_coefficients.shape[:-1], 3, num_bands
    ).transpose(-1, -2)
    # One matmul per channel, never a single [..., N, B] @ [..., B, 3]: on CUDA the fused form picks a different reduction and shifts the result by ~5e-7 off the Deep3DFaceRecon path this function is extracted from.
    return torch.cat(
        [
            basis @ coefficients[..., :1],
            basis @ coefficients[..., 1:2],
            basis @ coefficients[..., 2:],
        ],
        dim=-1,
    )
