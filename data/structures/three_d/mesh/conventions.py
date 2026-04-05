import torch

from data.structures.three_d.mesh.validate import (
    validate_mesh_uv_convention,
    validate_vertex_uv,
)


def transform_vertex_uv_convention(
    vertex_uv: torch.Tensor,
    source_convention: str,
    target_convention: str,
) -> torch.Tensor:
    """Transform one UV table between supported origin conventions.

    Args:
        vertex_uv: UV-coordinate tensor with shape `[U, 2]`.
        source_convention: Source UV-origin convention. `obj` means `v=0` is the
            bottom edge. `top_left` means `v=0` is the top edge.
        target_convention: Target UV-origin convention.

    Returns:
        UV-coordinate tensor in the target convention.
    """

    def _validate_inputs() -> None:
        """Validate UV-convention transform inputs.

        Args:
            None.

        Returns:
            None.
        """

        validate_vertex_uv(obj=vertex_uv)
        validate_mesh_uv_convention(convention=source_convention)
        validate_mesh_uv_convention(convention=target_convention)

    _validate_inputs()

    if source_convention == target_convention:
        return vertex_uv

    transformed_vertex_uv = vertex_uv.clone()
    transformed_vertex_uv[:, 1] = 1.0 - transformed_vertex_uv[:, 1]
    return transformed_vertex_uv.contiguous()
