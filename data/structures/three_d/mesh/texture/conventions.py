import torch


def transform_verts_uvs_convention(
    verts_uvs: torch.Tensor,
    source_convention: str,
    target_convention: str,
) -> torch.Tensor:
    """Transform one UV table between supported origin conventions.

    This is a pure transform. Input validation is owned by the validation
    modules and is intentionally not performed here.

    Args:
        verts_uvs: UV-coordinate tensor with shape `[U, 2]`.
        source_convention: Source UV-origin convention. `obj` means `v=0` is the
            bottom edge. `top_left` means `v=0` is the top edge.
        target_convention: Target UV-origin convention.

    Returns:
        UV-coordinate tensor in the target convention. The input tensor itself
        when the conventions match, otherwise a contiguous V-flipped copy.
    """

    if source_convention == target_convention:
        return verts_uvs

    flipped = verts_uvs.clone()
    flipped[:, 1] = 1.0 - flipped[:, 1]
    return flipped.contiguous()
