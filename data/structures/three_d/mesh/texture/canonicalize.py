"""OBJ-style vt layout <-> seam-safe canonical layout for `MeshTextureUVTextureMap`."""

from typing import List, Tuple

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
    face_corner_uvs = verts_uvs[faces_uvs_long]
    face_u = face_corner_uvs[..., 0]
    face_u_span = face_u.max(dim=1).values - face_u.min(dim=1).values
    seam_face_mask = face_u_span > 0.5

    if not bool(seam_face_mask.any().item()):
        return verts_uvs.contiguous(), faces_uvs_long

    seam_face_indices = torch.nonzero(seam_face_mask, as_tuple=False).reshape(-1)

    used_by_seam = torch.zeros(
        int(verts_uvs.shape[0]),
        dtype=torch.bool,
        device=verts_uvs.device,
    )
    used_by_non_seam = torch.zeros_like(used_by_seam)
    used_by_seam[faces_uvs_long[seam_face_indices].reshape(-1)] = True
    non_seam_face_mask = ~seam_face_mask
    if bool(non_seam_face_mask.any().item()):
        non_seam_face_indices = torch.nonzero(
            non_seam_face_mask, as_tuple=False
        ).reshape(-1)
        used_by_non_seam[faces_uvs_long[non_seam_face_indices].reshape(-1)] = True

    canonical_verts_uvs = verts_uvs.clone().contiguous()
    canonical_faces_uvs = faces_uvs_long.clone().contiguous()

    seam_only_row_mask = used_by_seam & ~used_by_non_seam
    if bool(seam_only_row_mask.any().item()):
        seam_only_rows = torch.nonzero(seam_only_row_mask, as_tuple=False).reshape(-1)
        small_u_mask_per_row = canonical_verts_uvs[seam_only_rows, 0] < 0.5
        rows_to_shift = seam_only_rows[small_u_mask_per_row]
        canonical_verts_uvs[rows_to_shift, 0] = (
            canonical_verts_uvs[rows_to_shift, 0] + 1.0
        )

    shared_row_mask = used_by_seam & used_by_non_seam
    if bool(shared_row_mask.any().item()):
        shared_rows = torch.nonzero(shared_row_mask, as_tuple=False).reshape(-1)
        new_rows_for_seam: List[int] = []
        for shared_row_index_item in shared_rows.tolist():
            shared_row_index = int(shared_row_index_item)
            shared_u = float(canonical_verts_uvs[shared_row_index, 0].item())
            if shared_u >= 0.5:
                continue
            new_row_index = int(canonical_verts_uvs.shape[0]) + len(new_rows_for_seam)
            new_rows_for_seam.append(shared_row_index)
            face_corner_uses_shared = (
                canonical_faces_uvs[seam_face_indices] == shared_row_index
            )
            face_corner_positions = torch.nonzero(
                face_corner_uses_shared, as_tuple=False
            )
            for position_row in face_corner_positions.tolist():
                seam_face_local_index = int(position_row[0])
                corner_index = int(position_row[1])
                seam_face_global_index = int(
                    seam_face_indices[seam_face_local_index].item()
                )
                canonical_faces_uvs[seam_face_global_index, corner_index] = new_row_index
        if new_rows_for_seam:
            new_rows_tensor = torch.empty(
                (len(new_rows_for_seam), 2),
                dtype=canonical_verts_uvs.dtype,
                device=canonical_verts_uvs.device,
            )
            for new_row_local_index, source_row_index in enumerate(new_rows_for_seam):
                new_rows_tensor[new_row_local_index, 0] = (
                    canonical_verts_uvs[source_row_index, 0] + 1.0
                )
                new_rows_tensor[new_row_local_index, 1] = canonical_verts_uvs[
                    source_row_index, 1
                ]
            canonical_verts_uvs = torch.cat(
                [canonical_verts_uvs, new_rows_tensor], dim=0
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
    unshifted_row_indices = torch.nonzero(unshifted_row_mask, as_tuple=False).reshape(-1)
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
