"""OBJ-style vt layout <-> seam-safe canonical layout for `MeshTextureUVTextureMap`."""

from typing import Tuple

import torch


def shift_seam_crossing_faces_to_seam_safe(
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shift seam-crossing UV faces into the seam-safe canonical chart, duplicating any source vt row shared between a seam-crossing and a non-seam face.

    Args:
        verts_uvs: Source UV-coordinate table `[U, 2]`, values in `[0, 1]`.
        faces_uvs: Face-to-UV index tensor `[F, 3]`.

    Returns:
        Canonical `(verts_uvs_canonical, faces_uvs_canonical)` with
        `U' >= U` and per-face u-span over
        `verts_uvs_canonical[faces_uvs_canonical[f]]` <= 0.5.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a `torch.Tensor`. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` shape `[U, 2]`. " f"{verts_uvs.shape=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a `torch.Tensor`. " f"{type(faces_uvs)=}"
        )
        assert faces_uvs.ndim == 2 and faces_uvs.shape[1] == 3, (
            "Expected `faces_uvs` shape `[F, 3]`. " f"{faces_uvs.shape=}"
        )

    _validate_inputs()

    faces_uvs_long = faces_uvs.to(dtype=torch.long).contiguous()
    face_corner_u = verts_uvs[faces_uvs_long, 0]

    sorted_u = face_corner_u.sort(dim=1).values
    interior_gaps = sorted_u[:, 1:] - sorted_u[:, :-1]
    wraparound_gap = sorted_u[:, 0] + 1.0 - sorted_u[:, -1]
    largest_interior_gap, largest_interior_position = interior_gaps.max(dim=1)
    seam_face_mask = largest_interior_gap > wraparound_gap

    if not bool(seam_face_mask.any().item()):
        return verts_uvs.contiguous(), faces_uvs_long

    cut_low_u = sorted_u.gather(
        dim=1, index=largest_interior_position.unsqueeze(1)
    ).squeeze(1)
    corner_shift_mask = seam_face_mask.unsqueeze(1) & (
        face_corner_u <= cut_low_u.unsqueeze(1)
    )

    n_rows = int(verts_uvs.shape[0])
    flat_rows = faces_uvs_long.reshape(-1)
    flat_shift = corner_shift_mask.reshape(-1)
    row_wants_shift = torch.zeros(n_rows, dtype=torch.bool, device=verts_uvs.device)
    row_wants_keep = torch.zeros(n_rows, dtype=torch.bool, device=verts_uvs.device)
    row_wants_shift[flat_rows[flat_shift]] = True
    row_wants_keep[flat_rows[~flat_shift]] = True

    canonical_verts_uvs = verts_uvs.clone().contiguous()
    canonical_faces_uvs = faces_uvs_long.clone().contiguous()

    shift_in_place_row_mask = row_wants_shift & ~row_wants_keep
    canonical_verts_uvs[shift_in_place_row_mask, 0] = (
        canonical_verts_uvs[shift_in_place_row_mask, 0] + 1.0
    )

    fork_row_mask = row_wants_shift & row_wants_keep
    if bool(fork_row_mask.any().item()):
        fork_row_indices = torch.nonzero(fork_row_mask, as_tuple=False).reshape(-1)
        forked_row_of = torch.full(
            (n_rows,), -1, dtype=torch.long, device=verts_uvs.device
        )
        forked_row_of[fork_row_indices] = torch.arange(
            n_rows,
            n_rows + int(fork_row_indices.shape[0]),
            device=verts_uvs.device,
        )
        forked_rows = canonical_verts_uvs[fork_row_indices].clone()
        forked_rows[:, 0] = forked_rows[:, 0] + 1.0
        repoint_mask = corner_shift_mask & fork_row_mask[canonical_faces_uvs]
        canonical_faces_uvs[repoint_mask] = forked_row_of[
            canonical_faces_uvs[repoint_mask]
        ]
        canonical_verts_uvs = torch.cat(
            [canonical_verts_uvs, forked_rows], dim=0
        ).contiguous()

    return canonical_verts_uvs, canonical_faces_uvs.contiguous()


def collapse_seam_shifted_uv_rows(
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collapse seam-shifted UV rows back to the OBJ vt structure by detecting (u, v) / (u - 1, v) sibling pairs and emitting one vt entry referenced by both face-corner indices.

    Args:
        verts_uvs: Canonical seam-safe UV-coordinate table `[U_canonical, 2]`.
        faces_uvs: Canonical face-to-UV index tensor `[F, 3]`.

    Returns:
        OBJ-form `(obj_vt_table, obj_faces_uvs)` with `U_obj <= U_canonical`
        and all UV values wrapped into `[0, 1)` along u.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a `torch.Tensor`. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` shape `[U, 2]`. " f"{verts_uvs.shape=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a `torch.Tensor`. " f"{type(faces_uvs)=}"
        )
        assert faces_uvs.ndim == 2 and faces_uvs.shape[1] == 3, (
            "Expected `faces_uvs` shape `[F, 3]`. " f"{faces_uvs.shape=}"
        )

    _validate_inputs()

    faces_uvs_long = faces_uvs.to(dtype=torch.long).contiguous()
    wrapped_verts_uvs = verts_uvs.clone().contiguous()
    wrapped_verts_uvs[:, 0] = wrapped_verts_uvs[:, 0] - torch.floor(
        wrapped_verts_uvs[:, 0]
    )

    shifted_row_mask = verts_uvs[:, 0] > 1.0
    if not bool(shifted_row_mask.any().item()):
        return wrapped_verts_uvs, faces_uvs_long

    canonical_to_obj_index = torch.arange(
        int(verts_uvs.shape[0]),
        dtype=torch.long,
        device=verts_uvs.device,
    )

    shifted_row_indices = torch.nonzero(shifted_row_mask, as_tuple=False).reshape(-1)
    unshifted_row_mask = ~shifted_row_mask
    unshifted_row_indices = torch.nonzero(unshifted_row_mask, as_tuple=False).reshape(
        -1
    )
    unshifted_uvs = wrapped_verts_uvs[unshifted_row_indices]

    redundant_row_mask = torch.zeros_like(shifted_row_mask)
    for shifted_row_index_item in shifted_row_indices.tolist():
        shifted_row_index = int(shifted_row_index_item)
        target_uv = wrapped_verts_uvs[shifted_row_index]
        sibling_match = torch.all(
            torch.isclose(
                unshifted_uvs,
                target_uv.unsqueeze(0),
                atol=1.0e-06,
                rtol=0.0,
            ),
            dim=1,
        )
        if bool(sibling_match.any().item()):
            sibling_local_index = int(
                torch.nonzero(sibling_match, as_tuple=False).reshape(-1)[0].item()
            )
            sibling_row_index = int(unshifted_row_indices[sibling_local_index].item())
            canonical_to_obj_index[shifted_row_index] = sibling_row_index
            redundant_row_mask[shifted_row_index] = True

    keep_row_mask = ~redundant_row_mask
    obj_to_canonical_keep = torch.nonzero(keep_row_mask, as_tuple=False).reshape(-1)
    new_obj_index_for_canonical = torch.full(
        (int(verts_uvs.shape[0]),),
        -1,
        dtype=torch.long,
        device=verts_uvs.device,
    )
    new_obj_index_for_canonical[obj_to_canonical_keep] = torch.arange(
        int(obj_to_canonical_keep.shape[0]),
        dtype=torch.long,
        device=verts_uvs.device,
    )
    final_canonical_to_obj = new_obj_index_for_canonical[canonical_to_obj_index]

    obj_vt_table = wrapped_verts_uvs[obj_to_canonical_keep].contiguous()
    obj_faces_uvs = final_canonical_to_obj[faces_uvs_long].contiguous()

    return obj_vt_table, obj_faces_uvs
