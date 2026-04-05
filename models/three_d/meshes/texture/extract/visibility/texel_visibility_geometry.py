"""Low-level texel-visibility geometry kernels."""

from typing import List, Tuple

import torch
import torch.nn.functional as F

TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET = 2**18


def _plan_multi_face_pixel_chunks(
    face_count_per_pixel: torch.Tensor,
    max_vertices_per_polygon: int,
    target_split_line_budget: int,
) -> List[Tuple[int, int]]:
    """Plan sorted multi-face pixel chunks under a split-line budget.

    Args:
        face_count_per_pixel: Sorted per-pixel face counts `[Np]`.
        max_vertices_per_polygon: Maximum padded polygon vertex capacity.
        target_split_line_budget: Maximum chunk budget in split-line units.

    Returns:
        List of half-open `(chunk_start, chunk_end)` ranges.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(face_count_per_pixel, torch.Tensor), (
            "Expected `face_count_per_pixel` to be a tensor. "
            f"Got {type(face_count_per_pixel)=}."
        )
        assert face_count_per_pixel.ndim == 1, (
            "Expected `face_count_per_pixel` to have shape `[Np]`. "
            f"Got {face_count_per_pixel.shape=}."
        )
        assert isinstance(max_vertices_per_polygon, int), (
            "Expected `max_vertices_per_polygon` to be an integer. "
            f"Got {type(max_vertices_per_polygon)=}."
        )
        assert max_vertices_per_polygon > 0, (
            "Expected `max_vertices_per_polygon` to be positive. "
            f"Got {max_vertices_per_polygon=}."
        )
        assert isinstance(target_split_line_budget, int), (
            "Expected `target_split_line_budget` to be an integer. "
            f"Got {type(target_split_line_budget)=}."
        )
        assert target_split_line_budget > 0, (
            "Expected `target_split_line_budget` to be positive. "
            f"Got {target_split_line_budget=}."
        )
        if face_count_per_pixel.numel() > 0:
            assert torch.all(face_count_per_pixel > 0), (
                "Expected every sorted pixel face count to be positive. "
                f"Got {face_count_per_pixel.min()=} {face_count_per_pixel.shape=}."
            )

    _validate_inputs()

    if face_count_per_pixel.numel() == 0:
        return []

    chunk_ranges: List[Tuple[int, int]] = []
    chunk_start = 0
    while chunk_start < face_count_per_pixel.shape[0]:
        chunk_end = chunk_start + 1
        while chunk_end < face_count_per_pixel.shape[0]:
            next_max_face_count = int(face_count_per_pixel[chunk_end].item())
            prospective_chunk_size = chunk_end - chunk_start + 1
            prospective_split_line_count = (
                next_max_face_count * max_vertices_per_polygon
                + (next_max_face_count * (next_max_face_count - 1)) // 2
            )
            prospective_budget = prospective_chunk_size * prospective_split_line_count
            if prospective_budget > target_split_line_budget:
                break
            chunk_end += 1
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunk_ranges


def _gather_visible_pixel_face_polygons(
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_indices: torch.Tensor,
    pixel_face_slot_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather selected pixel-face polygons into flat visible outputs.

    Args:
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major vertex counts [Np, M].
        pixel_face_indices: Pixel-major local face indices [Np, M].
        pixel_face_slot_mask: Pixel-major selection mask [Np, M].

    Returns:
        Tuple of:
            visible polygons [Pv, Vmax, 2],
            visible polygon vertex counts [Pv],
            visible local face indices [Pv].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(pixel_polygon_vertices, torch.Tensor), (
            "Expected `pixel_polygon_vertices` to be a tensor. "
            f"Got {type(pixel_polygon_vertices)=}."
        )
        assert isinstance(pixel_polygon_vertex_counts, torch.Tensor), (
            "Expected `pixel_polygon_vertex_counts` to be a tensor. "
            f"Got {type(pixel_polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_face_indices, torch.Tensor), (
            "Expected `pixel_face_indices` to be a tensor. "
            f"Got {type(pixel_face_indices)=}."
        )
        assert isinstance(pixel_face_slot_mask, torch.Tensor), (
            "Expected `pixel_face_slot_mask` to be a tensor. "
            f"Got {type(pixel_face_slot_mask)=}."
        )
        assert pixel_polygon_vertices.ndim == 4, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertices.shape[3] == 2, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertex_counts.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_polygon_vertex_counts` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_polygon_vertex_counts.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_indices.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_indices` to match the first two dimensions of "
            "`pixel_polygon_vertices`. "
            f"Got {pixel_face_indices.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_slot_mask.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_slot_mask` to match the first two dimensions of "
            "`pixel_polygon_vertices`. "
            f"Got {pixel_face_slot_mask.shape=} {pixel_polygon_vertices.shape=}."
        )

    _validate_inputs()

    return (
        pixel_polygon_vertices[pixel_face_slot_mask].contiguous(),
        pixel_polygon_vertex_counts[pixel_face_slot_mask].contiguous(),
        pixel_face_indices[pixel_face_slot_mask].contiguous(),
    )


def _compute_pair_positive_area_overlap_mask(
    first_polygon_vertices: torch.Tensor,
    first_polygon_vertex_counts: torch.Tensor,
    second_polygon_vertices: torch.Tensor,
    second_polygon_vertex_counts: torch.Tensor,
) -> torch.Tensor:
    """Detect positive-area overlap for convex polygon pairs.

    Args:
        first_polygon_vertices: First convex polygons [P, Vmax, 2].
        first_polygon_vertex_counts: First polygon vertex counts [P].
        second_polygon_vertices: Second convex polygons [P, Vmax, 2].
        second_polygon_vertex_counts: Second polygon vertex counts [P].

    Returns:
        Boolean positive-area overlap mask [P].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(first_polygon_vertices, torch.Tensor), (
            "Expected `first_polygon_vertices` to be a tensor. "
            f"Got {type(first_polygon_vertices)=}."
        )
        assert isinstance(first_polygon_vertex_counts, torch.Tensor), (
            "Expected `first_polygon_vertex_counts` to be a tensor. "
            f"Got {type(first_polygon_vertex_counts)=}."
        )
        assert isinstance(second_polygon_vertices, torch.Tensor), (
            "Expected `second_polygon_vertices` to be a tensor. "
            f"Got {type(second_polygon_vertices)=}."
        )
        assert isinstance(second_polygon_vertex_counts, torch.Tensor), (
            "Expected `second_polygon_vertex_counts` to be a tensor. "
            f"Got {type(second_polygon_vertex_counts)=}."
        )
        assert first_polygon_vertices.ndim == 3, (
            "Expected `first_polygon_vertices` to have shape `[P, Vmax, 2]`. "
            f"Got {first_polygon_vertices.shape=}."
        )
        assert first_polygon_vertices.shape[2] == 2, (
            "Expected `first_polygon_vertices` to have shape `[P, Vmax, 2]`. "
            f"Got {first_polygon_vertices.shape=}."
        )
        assert second_polygon_vertices.shape == first_polygon_vertices.shape, (
            "Expected `second_polygon_vertices` to match `first_polygon_vertices`. "
            f"Got {second_polygon_vertices.shape=} {first_polygon_vertices.shape=}."
        )
        assert first_polygon_vertex_counts.shape == (
            first_polygon_vertices.shape[0],
        ), (
            "Expected `first_polygon_vertex_counts` to have shape `[P]`. "
            f"Got {first_polygon_vertex_counts.shape=} {first_polygon_vertices.shape=}."
        )
        assert (
            second_polygon_vertex_counts.shape == first_polygon_vertex_counts.shape
        ), (
            "Expected `second_polygon_vertex_counts` to match "
            "`first_polygon_vertex_counts`. "
            f"Got {second_polygon_vertex_counts.shape=} "
            f"{first_polygon_vertex_counts.shape=}."
        )

    _validate_inputs()

    pair_count = first_polygon_vertices.shape[0]
    if pair_count == 0:
        return torch.zeros(
            (0,),
            device=first_polygon_vertices.device,
            dtype=torch.bool,
        )

    max_vertices_per_polygon = first_polygon_vertices.shape[1]
    vertex_indices = torch.arange(
        max_vertices_per_polygon,
        device=first_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1)
    first_vertex_valid_mask = vertex_indices < first_polygon_vertex_counts.reshape(
        -1, 1
    )
    second_vertex_valid_mask = vertex_indices < second_polygon_vertex_counts.reshape(
        -1,
        1,
    )
    positive_infinity = torch.full(
        (1,),
        fill_value=torch.inf,
        device=first_polygon_vertices.device,
        dtype=first_polygon_vertices.dtype,
    )
    negative_infinity = torch.full(
        (1,),
        fill_value=-torch.inf,
        device=first_polygon_vertices.device,
        dtype=first_polygon_vertices.dtype,
    )

    first_min_x = torch.where(
        first_vertex_valid_mask,
        first_polygon_vertices[:, :, 0],
        positive_infinity,
    ).amin(dim=1)
    first_max_x = torch.where(
        first_vertex_valid_mask,
        first_polygon_vertices[:, :, 0],
        negative_infinity,
    ).amax(dim=1)
    first_min_y = torch.where(
        first_vertex_valid_mask,
        first_polygon_vertices[:, :, 1],
        positive_infinity,
    ).amin(dim=1)
    first_max_y = torch.where(
        first_vertex_valid_mask,
        first_polygon_vertices[:, :, 1],
        negative_infinity,
    ).amax(dim=1)
    second_min_x = torch.where(
        second_vertex_valid_mask,
        second_polygon_vertices[:, :, 0],
        positive_infinity,
    ).amin(dim=1)
    second_max_x = torch.where(
        second_vertex_valid_mask,
        second_polygon_vertices[:, :, 0],
        negative_infinity,
    ).amax(dim=1)
    second_min_y = torch.where(
        second_vertex_valid_mask,
        second_polygon_vertices[:, :, 1],
        positive_infinity,
    ).amin(dim=1)
    second_max_y = torch.where(
        second_vertex_valid_mask,
        second_polygon_vertices[:, :, 1],
        negative_infinity,
    ).amax(dim=1)
    bbox_overlap_mask = (
        (first_min_x < second_max_x)
        & (second_min_x < first_max_x)
        & (first_min_y < second_max_y)
        & (second_min_y < first_max_y)
    )
    positive_area_overlap_mask = torch.zeros(
        (pair_count,),
        device=first_polygon_vertices.device,
        dtype=torch.bool,
    )
    if not torch.any(bbox_overlap_mask):
        return positive_area_overlap_mask.contiguous()

    overlapping_first_polygon_vertices = first_polygon_vertices[bbox_overlap_mask]
    overlapping_first_polygon_vertex_counts = first_polygon_vertex_counts[
        bbox_overlap_mask
    ]
    overlapping_second_polygon_vertices = second_polygon_vertices[bbox_overlap_mask]
    overlapping_second_polygon_vertex_counts = second_polygon_vertex_counts[
        bbox_overlap_mask
    ]
    batch_indices = torch.arange(
        overlapping_first_polygon_vertices.shape[0],
        device=first_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    edge_indices = torch.arange(
        max_vertices_per_polygon,
        device=first_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1)
    first_next_edge_indices = torch.where(
        edge_indices + 1 < overlapping_first_polygon_vertex_counts.reshape(-1, 1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    first_edge_valid_mask = edge_indices < (
        overlapping_first_polygon_vertex_counts.reshape(-1, 1)
    )
    first_edge_direction = (
        overlapping_first_polygon_vertices[batch_indices, first_next_edge_indices]
        - overlapping_first_polygon_vertices
    )
    first_edge_axes = torch.stack(
        [-first_edge_direction[..., 1], first_edge_direction[..., 0]],
        dim=2,
    )
    second_next_edge_indices = torch.where(
        edge_indices + 1 < overlapping_second_polygon_vertex_counts.reshape(-1, 1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    second_edge_valid_mask = edge_indices < (
        overlapping_second_polygon_vertex_counts.reshape(-1, 1)
    )
    second_edge_direction = (
        overlapping_second_polygon_vertices[batch_indices, second_next_edge_indices]
        - overlapping_second_polygon_vertices
    )
    second_edge_axes = torch.stack(
        [-second_edge_direction[..., 1], second_edge_direction[..., 0]],
        dim=2,
    )
    candidate_axes = torch.cat([first_edge_axes, second_edge_axes], dim=1)
    candidate_axis_valid_mask = torch.cat(
        [first_edge_valid_mask, second_edge_valid_mask],
        dim=1,
    ) & (torch.linalg.norm(candidate_axes, dim=2) > 1.0e-12)
    first_projection = torch.sum(
        overlapping_first_polygon_vertices.unsqueeze(2) * candidate_axes.unsqueeze(1),
        dim=3,
    )
    second_projection = torch.sum(
        overlapping_second_polygon_vertices.unsqueeze(2) * candidate_axes.unsqueeze(1),
        dim=3,
    )
    first_projection_min = torch.where(
        first_vertex_valid_mask[bbox_overlap_mask].unsqueeze(2),
        first_projection,
        positive_infinity,
    ).amin(dim=1)
    first_projection_max = torch.where(
        first_vertex_valid_mask[bbox_overlap_mask].unsqueeze(2),
        first_projection,
        negative_infinity,
    ).amax(dim=1)
    second_projection_min = torch.where(
        second_vertex_valid_mask[bbox_overlap_mask].unsqueeze(2),
        second_projection,
        positive_infinity,
    ).amin(dim=1)
    second_projection_max = torch.where(
        second_vertex_valid_mask[bbox_overlap_mask].unsqueeze(2),
        second_projection,
        negative_infinity,
    ).amax(dim=1)
    separating_axis_mask = candidate_axis_valid_mask & (
        (first_projection_max <= second_projection_min + 1.0e-12)
        | (second_projection_max <= first_projection_min + 1.0e-12)
    )
    positive_area_overlap_mask[bbox_overlap_mask] = ~torch.any(
        separating_axis_mask,
        dim=1,
    )
    return positive_area_overlap_mask.contiguous()


def _compute_triangle_pixel_square_positive_area_overlap_mask(
    triangle_vertices: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
) -> torch.Tensor:
    """Detect positive-area overlap between triangles and pixel squares.

    Args:
        triangle_vertices: Triangle vertices [N, 3, 2].
        pixel_x: Pixel-center x coordinate for each triangle [N].
        pixel_y: Pixel-center y coordinate for each triangle [N].

    Returns:
        Boolean positive-area overlap mask [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(triangle_vertices, torch.Tensor), (
            "Expected `triangle_vertices` to be a tensor. "
            f"Got {type(triangle_vertices)=}."
        )
        assert isinstance(pixel_x, torch.Tensor), (
            "Expected `pixel_x` to be a tensor. " f"Got {type(pixel_x)=}."
        )
        assert isinstance(pixel_y, torch.Tensor), (
            "Expected `pixel_y` to be a tensor. " f"Got {type(pixel_y)=}."
        )
        assert triangle_vertices.ndim == 3, (
            "Expected `triangle_vertices` to have shape `[N, 3, 2]`. "
            f"Got {triangle_vertices.shape=}."
        )
        assert triangle_vertices.shape[1:] == (3, 2), (
            "Expected `triangle_vertices` to have shape `[N, 3, 2]`. "
            f"Got {triangle_vertices.shape=}."
        )
        assert pixel_x.shape == (triangle_vertices.shape[0],), (
            "Expected one pixel x coordinate per triangle. "
            f"Got {pixel_x.shape=} {triangle_vertices.shape=}."
        )
        assert pixel_y.shape == (triangle_vertices.shape[0],), (
            "Expected one pixel y coordinate per triangle. "
            f"Got {pixel_y.shape=} {triangle_vertices.shape=}."
        )
        assert triangle_vertices.device == pixel_x.device, (
            "Expected `triangle_vertices` and `pixel_x` to share a device. "
            f"Got {triangle_vertices.device=} {pixel_x.device=}."
        )
        assert triangle_vertices.device == pixel_y.device, (
            "Expected `triangle_vertices` and `pixel_y` to share a device. "
            f"Got {triangle_vertices.device=} {pixel_y.device=}."
        )

    _validate_inputs()

    triangle_count = triangle_vertices.shape[0]
    if triangle_count == 0:
        return torch.zeros(
            (0,),
            device=triangle_vertices.device,
            dtype=torch.bool,
        )

    xmin = pixel_x - 0.5
    xmax = pixel_x + 0.5
    ymin = pixel_y - 0.5
    ymax = pixel_y + 0.5
    triangle_min_x = triangle_vertices[:, :, 0].amin(dim=1)
    triangle_max_x = triangle_vertices[:, :, 0].amax(dim=1)
    triangle_min_y = triangle_vertices[:, :, 1].amin(dim=1)
    triangle_max_y = triangle_vertices[:, :, 1].amax(dim=1)
    bbox_overlap_mask = (
        (triangle_min_x < xmax)
        & (xmin < triangle_max_x)
        & (triangle_min_y < ymax)
        & (ymin < triangle_max_y)
    )
    positive_area_overlap_mask = torch.zeros(
        (triangle_count,),
        device=triangle_vertices.device,
        dtype=torch.bool,
    )
    if not torch.any(bbox_overlap_mask):
        return positive_area_overlap_mask.contiguous()

    clipped_polygon_vertices, clipped_polygon_vertex_counts = (
        _clip_triangle_polygons_to_pixel_squares(
            triangle_vertices=triangle_vertices[bbox_overlap_mask],
            pixel_x=pixel_x[bbox_overlap_mask],
            pixel_y=pixel_y[bbox_overlap_mask],
            output_vertex_capacity=8,
        )
    )
    clipped_polygon_area = _compute_convex_polygon_areas(
        polygon_vertices=clipped_polygon_vertices,
        polygon_vertex_counts=clipped_polygon_vertex_counts,
    )
    positive_area_overlap_mask[bbox_overlap_mask] = (
        clipped_polygon_vertex_counts >= 3
    ) & (clipped_polygon_area > 1.0e-12)
    return positive_area_overlap_mask.contiguous()


def _compute_multi_face_pixel_second_bucket_mask(
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Detect which multi-face pixels require full overlap resolution.

    Args:
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major vertex counts [Np, M].
        pixel_face_valid_mask: Pixel-major face validity mask [Np, M].

    Returns:
        Boolean second-bucket mask [Np].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(pixel_polygon_vertices, torch.Tensor), (
            "Expected `pixel_polygon_vertices` to be a tensor. "
            f"Got {type(pixel_polygon_vertices)=}."
        )
        assert isinstance(pixel_polygon_vertex_counts, torch.Tensor), (
            "Expected `pixel_polygon_vertex_counts` to be a tensor. "
            f"Got {type(pixel_polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_face_valid_mask, torch.Tensor), (
            "Expected `pixel_face_valid_mask` to be a tensor. "
            f"Got {type(pixel_face_valid_mask)=}."
        )
        assert pixel_polygon_vertices.ndim == 4, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertices.shape[3] == 2, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertex_counts.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_polygon_vertex_counts` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_polygon_vertex_counts.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_valid_mask.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_valid_mask` to match the first two dimensions of "
            "`pixel_polygon_vertices`. "
            f"Got {pixel_face_valid_mask.shape=} {pixel_polygon_vertices.shape=}."
        )

    _validate_inputs()

    pixel_count, max_faces_per_pixel = pixel_polygon_vertices.shape[:2]
    pair_first_indices, pair_second_indices = torch.triu_indices(
        max_faces_per_pixel,
        max_faces_per_pixel,
        offset=1,
        device=pixel_polygon_vertices.device,
    )
    if pair_first_indices.numel() == 0:
        return torch.zeros(
            (pixel_count,),
            device=pixel_polygon_vertices.device,
            dtype=torch.bool,
        )

    pair_valid_mask = (
        pixel_face_valid_mask[:, pair_first_indices]
        & pixel_face_valid_mask[
            :,
            pair_second_indices,
        ]
    )
    if not torch.any(pair_valid_mask):
        return torch.zeros(
            (pixel_count,),
            device=pixel_polygon_vertices.device,
            dtype=torch.bool,
        )

    first_pair_polygon_vertices = pixel_polygon_vertices[:, pair_first_indices][
        pair_valid_mask
    ]
    first_pair_polygon_vertex_counts = pixel_polygon_vertex_counts[
        :, pair_first_indices
    ][pair_valid_mask]
    second_pair_polygon_vertices = pixel_polygon_vertices[:, pair_second_indices][
        pair_valid_mask
    ]
    second_pair_polygon_vertex_counts = pixel_polygon_vertex_counts[
        :, pair_second_indices
    ][pair_valid_mask]
    positive_area_overlap_mask = _compute_pair_positive_area_overlap_mask(
        first_polygon_vertices=first_pair_polygon_vertices,
        first_polygon_vertex_counts=first_pair_polygon_vertex_counts,
        second_polygon_vertices=second_pair_polygon_vertices,
        second_polygon_vertex_counts=second_pair_polygon_vertex_counts,
    )
    second_bucket_mask = torch.zeros(
        (pixel_count,),
        device=pixel_polygon_vertices.device,
        dtype=torch.bool,
    )
    if torch.any(positive_area_overlap_mask):
        pair_pixel_indices = torch.nonzero(pair_valid_mask, as_tuple=False)[:, 0]
        second_bucket_mask[pair_pixel_indices[positive_area_overlap_mask]] = True
    return second_bucket_mask.contiguous()


def _build_visible_face_pixel_polygons(
    clipped_polygon_vertices: torch.Tensor,
    clipped_polygon_vertex_counts: torch.Tensor,
    clipped_pixel_indices: torch.Tensor,
    clipped_face_indices: torch.Tensor,
    face_inverse_depth_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact visible face-pixel polygons in batched tensor form.

    Args:
        clipped_polygon_vertices: Face-pixel polygons [P, Vmax, 2].
        clipped_polygon_vertex_counts: Valid polygon vertex counts [P].
        clipped_pixel_indices: Pixel indices [P, 2] in `(y, x)` order.
        clipped_face_indices: Local face indices [P].
        face_inverse_depth_coefficients: Inverse-depth coefficients [F, 3].

    Returns:
        Tuple of:
            visible polygons [Pv, Vvis, 2],
            visible polygon vertex counts [Pv],
            local face indices [Pv].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(clipped_polygon_vertices, torch.Tensor), (
            "Expected `clipped_polygon_vertices` to be a tensor. "
            f"Got {type(clipped_polygon_vertices)=}."
        )
        assert isinstance(clipped_polygon_vertex_counts, torch.Tensor), (
            "Expected `clipped_polygon_vertex_counts` to be a tensor. "
            f"Got {type(clipped_polygon_vertex_counts)=}."
        )
        assert isinstance(clipped_pixel_indices, torch.Tensor), (
            "Expected `clipped_pixel_indices` to be a tensor. "
            f"Got {type(clipped_pixel_indices)=}."
        )
        assert isinstance(clipped_face_indices, torch.Tensor), (
            "Expected `clipped_face_indices` to be a tensor. "
            f"Got {type(clipped_face_indices)=}."
        )
        assert isinstance(face_inverse_depth_coefficients, torch.Tensor), (
            "Expected `face_inverse_depth_coefficients` to be a tensor. "
            f"Got {type(face_inverse_depth_coefficients)=}."
        )
        assert clipped_polygon_vertices.ndim == 3, (
            "Expected `clipped_polygon_vertices` to have shape `[P, Vmax, 2]`. "
            f"Got {clipped_polygon_vertices.shape=}."
        )
        assert clipped_polygon_vertices.shape[2] == 2, (
            "Expected `clipped_polygon_vertices` to have shape `[P, Vmax, 2]`. "
            f"Got {clipped_polygon_vertices.shape=}."
        )
        assert clipped_polygon_vertex_counts.shape == (
            clipped_polygon_vertices.shape[0],
        ), (
            "Expected `clipped_polygon_vertex_counts` to have shape `[P]`. "
            f"Got {clipped_polygon_vertex_counts.shape=} "
            f"{clipped_polygon_vertices.shape=}."
        )
        assert clipped_pixel_indices.shape == (
            clipped_polygon_vertices.shape[0],
            2,
        ), (
            "Expected `clipped_pixel_indices` to have shape `[P, 2]`. "
            f"Got {clipped_pixel_indices.shape=} {clipped_polygon_vertices.shape=}."
        )
        assert clipped_face_indices.shape == (clipped_polygon_vertices.shape[0],), (
            "Expected `clipped_face_indices` to have shape `[P]`. "
            f"Got {clipped_face_indices.shape=} {clipped_polygon_vertices.shape=}."
        )
        assert face_inverse_depth_coefficients.ndim == 2, (
            "Expected `face_inverse_depth_coefficients` to have shape `[F, 3]`. "
            f"Got {face_inverse_depth_coefficients.shape=}."
        )
        assert face_inverse_depth_coefficients.shape[1] == 3, (
            "Expected `face_inverse_depth_coefficients` to have shape `[F, 3]`. "
            f"Got {face_inverse_depth_coefficients.shape=}."
        )

    _validate_inputs()

    if clipped_polygon_vertices.shape[0] == 0:
        return (
            torch.zeros(
                (0, clipped_polygon_vertices.shape[1], 2),
                device=clipped_polygon_vertices.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=clipped_polygon_vertices.device,
                dtype=torch.long,
            ),
            torch.zeros(
                (0,),
                device=clipped_polygon_vertices.device,
                dtype=torch.long,
            ),
        )

    (
        pixel_indices,
        pixel_polygon_vertices,
        pixel_polygon_vertex_counts,
        pixel_face_indices,
        pixel_face_valid_mask,
        pixel_inverse_depth_coefficients,
    ) = _pack_face_pixel_polygons_by_pixel(
        clipped_polygon_vertices=clipped_polygon_vertices,
        clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
        clipped_pixel_indices=clipped_pixel_indices,
        clipped_face_indices=clipped_face_indices,
        face_inverse_depth_coefficients=face_inverse_depth_coefficients,
    )
    face_count_per_pixel = pixel_face_valid_mask.sum(dim=1)
    single_face_pixel_mask = face_count_per_pixel == 1
    multi_face_pixel_mask = face_count_per_pixel > 1

    visible_polygon_vertices_chunks: List[torch.Tensor] = []
    visible_polygon_vertex_counts_chunks: List[torch.Tensor] = []
    visible_polygon_face_indices_chunks: List[torch.Tensor] = []
    if torch.any(single_face_pixel_mask):
        (
            single_face_visible_polygon_vertices,
            single_face_visible_polygon_vertex_counts,
            single_face_visible_polygon_face_indices,
        ) = _gather_visible_pixel_face_polygons(
            pixel_polygon_vertices=pixel_polygon_vertices[single_face_pixel_mask],
            pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[
                single_face_pixel_mask
            ],
            pixel_face_indices=pixel_face_indices[single_face_pixel_mask],
            pixel_face_slot_mask=pixel_face_valid_mask[single_face_pixel_mask],
        )
        visible_polygon_vertices_chunks.append(single_face_visible_polygon_vertices)
        visible_polygon_vertex_counts_chunks.append(
            single_face_visible_polygon_vertex_counts
        )
        visible_polygon_face_indices_chunks.append(
            single_face_visible_polygon_face_indices
        )
    if not torch.any(multi_face_pixel_mask):
        return (
            torch.cat(visible_polygon_vertices_chunks, dim=0).contiguous(),
            torch.cat(visible_polygon_vertex_counts_chunks, dim=0).contiguous(),
            torch.cat(visible_polygon_face_indices_chunks, dim=0).contiguous(),
        )

    (
        multi_face_visible_polygon_vertices,
        multi_face_visible_polygon_vertex_counts,
        multi_face_visible_polygon_face_indices,
    ) = _build_visible_multi_face_pixel_polygons(
        pixel_indices=pixel_indices[multi_face_pixel_mask],
        pixel_polygon_vertices=pixel_polygon_vertices[multi_face_pixel_mask],
        pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[multi_face_pixel_mask],
        pixel_face_indices=pixel_face_indices[multi_face_pixel_mask],
        pixel_face_valid_mask=pixel_face_valid_mask[multi_face_pixel_mask],
        pixel_inverse_depth_coefficients=pixel_inverse_depth_coefficients[
            multi_face_pixel_mask
        ],
    )
    visible_polygon_vertices_chunks.append(multi_face_visible_polygon_vertices)
    visible_polygon_vertex_counts_chunks.append(
        multi_face_visible_polygon_vertex_counts
    )
    visible_polygon_face_indices_chunks.append(multi_face_visible_polygon_face_indices)
    max_visible_polygon_vertex_capacity = max(
        polygon_vertices.shape[1]
        for polygon_vertices in visible_polygon_vertices_chunks
    )
    visible_polygon_vertices_chunks = [
        (
            polygon_vertices
            if polygon_vertices.shape[1] == max_visible_polygon_vertex_capacity
            else F.pad(
                polygon_vertices,
                pad=(
                    0,
                    0,
                    0,
                    max_visible_polygon_vertex_capacity - polygon_vertices.shape[1],
                ),
            )
        )
        for polygon_vertices in visible_polygon_vertices_chunks
    ]
    return (
        torch.cat(visible_polygon_vertices_chunks, dim=0).contiguous(),
        torch.cat(visible_polygon_vertex_counts_chunks, dim=0).contiguous(),
        torch.cat(visible_polygon_face_indices_chunks, dim=0).contiguous(),
    )


def _build_visible_multi_face_pixel_polygons(
    pixel_indices: torch.Tensor,
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_indices: torch.Tensor,
    pixel_face_valid_mask: torch.Tensor,
    pixel_inverse_depth_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve visible polygons for the multi-face pixels in memory-bounded batches.

    Args:
        pixel_indices: Pixel indices [Np, 2] in `(y, x)` order.
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major vertex counts [Np, M].
        pixel_face_indices: Pixel-major local face indices [Np, M].
        pixel_face_valid_mask: Pixel-major face validity mask [Np, M].
        pixel_inverse_depth_coefficients: Pixel-major inverse-depth coefficients [Np, M, 3].

    Returns:
        Tuple of:
            visible polygons [Pv, Vvis, 2],
            visible polygon vertex counts [Pv],
            visible local face indices [Pv].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(pixel_indices, torch.Tensor), (
            "Expected `pixel_indices` to be a tensor. " f"Got {type(pixel_indices)=}."
        )
        assert isinstance(pixel_polygon_vertices, torch.Tensor), (
            "Expected `pixel_polygon_vertices` to be a tensor. "
            f"Got {type(pixel_polygon_vertices)=}."
        )
        assert isinstance(pixel_polygon_vertex_counts, torch.Tensor), (
            "Expected `pixel_polygon_vertex_counts` to be a tensor. "
            f"Got {type(pixel_polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_face_indices, torch.Tensor), (
            "Expected `pixel_face_indices` to be a tensor. "
            f"Got {type(pixel_face_indices)=}."
        )
        assert isinstance(pixel_face_valid_mask, torch.Tensor), (
            "Expected `pixel_face_valid_mask` to be a tensor. "
            f"Got {type(pixel_face_valid_mask)=}."
        )
        assert isinstance(pixel_inverse_depth_coefficients, torch.Tensor), (
            "Expected `pixel_inverse_depth_coefficients` to be a tensor. "
            f"Got {type(pixel_inverse_depth_coefficients)=}."
        )
        assert pixel_indices.ndim == 2, (
            "Expected `pixel_indices` to have shape `[Np, 2]`. "
            f"Got {pixel_indices.shape=}."
        )
        assert pixel_indices.shape[1] == 2, (
            "Expected `pixel_indices` to have shape `[Np, 2]`. "
            f"Got {pixel_indices.shape=}."
        )
        assert pixel_polygon_vertices.ndim == 4, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertices.shape[3] == 2, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertex_counts.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_polygon_vertex_counts` to match the first two "
            "dimensions of `pixel_polygon_vertices`. "
            f"Got {pixel_polygon_vertex_counts.shape=} "
            f"{pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_indices.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_indices` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_face_indices.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_valid_mask.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_valid_mask` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_face_valid_mask.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_inverse_depth_coefficients.shape == (
            pixel_polygon_vertices.shape[0],
            pixel_polygon_vertices.shape[1],
            3,
        ), (
            "Expected `pixel_inverse_depth_coefficients` to have shape `[Np, M, 3]`. "
            f"Got {pixel_inverse_depth_coefficients.shape=} "
            f"{pixel_polygon_vertices.shape=}."
        )

    _validate_inputs()

    if pixel_indices.shape[0] == 0:
        return (
            torch.zeros(
                (0, pixel_polygon_vertices.shape[2], 2),
                device=pixel_polygon_vertices.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=pixel_polygon_vertices.device,
                dtype=torch.long,
            ),
            torch.zeros(
                (0,),
                device=pixel_polygon_vertices.device,
                dtype=torch.long,
            ),
        )

    face_count_per_pixel = pixel_face_valid_mask.sum(dim=1)
    sorted_pixel_indices = torch.argsort(face_count_per_pixel)
    face_count_per_pixel = face_count_per_pixel[sorted_pixel_indices]
    pixel_indices = pixel_indices[sorted_pixel_indices]
    pixel_polygon_vertices = pixel_polygon_vertices[sorted_pixel_indices]
    pixel_polygon_vertex_counts = pixel_polygon_vertex_counts[sorted_pixel_indices]
    pixel_face_indices = pixel_face_indices[sorted_pixel_indices]
    pixel_face_valid_mask = pixel_face_valid_mask[sorted_pixel_indices]
    pixel_inverse_depth_coefficients = pixel_inverse_depth_coefficients[
        sorted_pixel_indices
    ]

    visible_polygon_vertices_chunks: List[torch.Tensor] = []
    visible_polygon_vertex_counts_chunks: List[torch.Tensor] = []
    visible_polygon_face_indices_chunks: List[torch.Tensor] = []
    second_bucket_mask = _compute_multi_face_pixel_second_bucket_mask(
        pixel_polygon_vertices=pixel_polygon_vertices,
        pixel_polygon_vertex_counts=pixel_polygon_vertex_counts,
        pixel_face_valid_mask=pixel_face_valid_mask,
    )
    first_bucket_mask = ~second_bucket_mask
    if torch.any(first_bucket_mask):
        (
            first_bucket_visible_polygon_vertices,
            first_bucket_visible_polygon_vertex_counts,
            first_bucket_visible_polygon_face_indices,
        ) = _gather_visible_pixel_face_polygons(
            pixel_polygon_vertices=pixel_polygon_vertices[first_bucket_mask],
            pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[first_bucket_mask],
            pixel_face_indices=pixel_face_indices[first_bucket_mask],
            pixel_face_slot_mask=pixel_face_valid_mask[first_bucket_mask],
        )
        visible_polygon_vertices_chunks.append(first_bucket_visible_polygon_vertices)
        visible_polygon_vertex_counts_chunks.append(
            first_bucket_visible_polygon_vertex_counts
        )
        visible_polygon_face_indices_chunks.append(
            first_bucket_visible_polygon_face_indices
        )
    if not torch.any(second_bucket_mask):
        return (
            torch.cat(visible_polygon_vertices_chunks, dim=0).contiguous(),
            torch.cat(visible_polygon_vertex_counts_chunks, dim=0).contiguous(),
            torch.cat(visible_polygon_face_indices_chunks, dim=0).contiguous(),
        )

    pixel_indices = pixel_indices[second_bucket_mask]
    face_count_per_pixel = face_count_per_pixel[second_bucket_mask]
    pixel_polygon_vertices = pixel_polygon_vertices[second_bucket_mask]
    pixel_polygon_vertex_counts = pixel_polygon_vertex_counts[second_bucket_mask]
    pixel_face_indices = pixel_face_indices[second_bucket_mask]
    pixel_face_valid_mask = pixel_face_valid_mask[second_bucket_mask]
    pixel_inverse_depth_coefficients = pixel_inverse_depth_coefficients[
        second_bucket_mask
    ]
    max_vertices_per_polygon = pixel_polygon_vertices.shape[2]
    chunk_ranges = _plan_multi_face_pixel_chunks(
        face_count_per_pixel=face_count_per_pixel,
        max_vertices_per_polygon=max_vertices_per_polygon,
        target_split_line_budget=TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET,
    )
    for chunk_start, chunk_end in chunk_ranges:
        pixel_split_line_coefficients, pixel_split_line_valid_mask = (
            _build_padded_pixel_split_line_coefficients(
                pixel_indices=pixel_indices[chunk_start:chunk_end],
                pixel_polygon_vertices=pixel_polygon_vertices[chunk_start:chunk_end],
                pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[
                    chunk_start:chunk_end
                ],
                pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end],
            )
        )
        (
            cell_polygon_vertices,
            cell_polygon_vertex_counts,
            cell_pixel_indices,
        ) = _build_batched_pixel_cell_polygons(
            pixel_indices=pixel_indices[chunk_start:chunk_end],
            pixel_polygon_vertices=pixel_polygon_vertices[chunk_start:chunk_end],
            pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[
                chunk_start:chunk_end
            ],
            pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end],
            pixel_split_line_coefficients=pixel_split_line_coefficients,
            pixel_split_line_valid_mask=pixel_split_line_valid_mask,
        )
        (
            chunk_visible_polygon_vertices,
            chunk_visible_polygon_vertex_counts,
            chunk_visible_polygon_face_indices,
        ) = _assign_visible_faces_to_cells(
            cell_polygon_vertices=cell_polygon_vertices,
            cell_polygon_vertex_counts=cell_polygon_vertex_counts,
            cell_pixel_indices=cell_pixel_indices,
            pixel_polygon_vertices=pixel_polygon_vertices[chunk_start:chunk_end],
            pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[
                chunk_start:chunk_end
            ],
            pixel_face_indices=pixel_face_indices[chunk_start:chunk_end],
            pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end],
            pixel_inverse_depth_coefficients=pixel_inverse_depth_coefficients[
                chunk_start:chunk_end
            ],
        )
        visible_polygon_vertices_chunks.append(chunk_visible_polygon_vertices)
        visible_polygon_vertex_counts_chunks.append(chunk_visible_polygon_vertex_counts)
        visible_polygon_face_indices_chunks.append(chunk_visible_polygon_face_indices)

    max_visible_polygon_vertex_capacity = max(
        polygon_vertices.shape[1]
        for polygon_vertices in visible_polygon_vertices_chunks
    )
    visible_polygon_vertices_chunks = [
        (
            polygon_vertices
            if polygon_vertices.shape[1] == max_visible_polygon_vertex_capacity
            else F.pad(
                polygon_vertices,
                pad=(
                    0,
                    0,
                    0,
                    max_visible_polygon_vertex_capacity - polygon_vertices.shape[1],
                ),
            )
        )
        for polygon_vertices in visible_polygon_vertices_chunks
    ]
    return (
        torch.cat(visible_polygon_vertices_chunks, dim=0).contiguous(),
        torch.cat(visible_polygon_vertex_counts_chunks, dim=0).contiguous(),
        torch.cat(visible_polygon_face_indices_chunks, dim=0).contiguous(),
    )


def _deduplicate_padded_pixel_split_lines(
    pixel_split_line_coefficients: torch.Tensor,
    pixel_split_line_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate canonical split lines independently within each pixel.

    Args:
        pixel_split_line_coefficients: Pixel-major split lines [Np, Lmax, 3].
        pixel_split_line_valid_mask: Pixel-major split-line validity mask [Np, Lmax].

    Returns:
        Tuple of:
            deduplicated split-line coefficients [Np, Lmax, 3],
            deduplicated split-line validity mask [Np, Lmax].
    """
    # Input validations
    assert isinstance(pixel_split_line_coefficients, torch.Tensor), (
        "Expected `pixel_split_line_coefficients` to be a tensor. "
        f"Got {type(pixel_split_line_coefficients)=}."
    )
    assert isinstance(pixel_split_line_valid_mask, torch.Tensor), (
        "Expected `pixel_split_line_valid_mask` to be a tensor. "
        f"Got {type(pixel_split_line_valid_mask)=}."
    )
    assert pixel_split_line_coefficients.ndim == 3, (
        "Expected `pixel_split_line_coefficients` to have shape `[Np, Lmax, 3]`. "
        f"Got {pixel_split_line_coefficients.shape=}."
    )
    assert pixel_split_line_coefficients.shape[2] == 3, (
        "Expected `pixel_split_line_coefficients` to have shape `[Np, Lmax, 3]`. "
        f"Got {pixel_split_line_coefficients.shape=}."
    )
    assert (
        pixel_split_line_valid_mask.shape == pixel_split_line_coefficients.shape[:2]
    ), (
        "Expected `pixel_split_line_valid_mask` to match the first two dimensions "
        "of `pixel_split_line_coefficients`. "
        f"Got {pixel_split_line_valid_mask.shape=} "
        f"{pixel_split_line_coefficients.shape=}."
    )

    if not torch.any(pixel_split_line_valid_mask):
        return (
            torch.zeros_like(pixel_split_line_coefficients),
            torch.zeros_like(pixel_split_line_valid_mask),
        )

    pixel_count, max_split_line_count = pixel_split_line_valid_mask.shape
    assert pixel_count < (1 << 24), (
        "Expected `pixel_count` to stay below the exact-float32 integer range "
        "for batched split-line deduplication. "
        f"{pixel_count=}"
    )
    pixel_row_indices = torch.arange(
        pixel_count,
        device=pixel_split_line_coefficients.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    flat_pixel_indices = pixel_row_indices.expand(
        -1,
        max_split_line_count,
    )[pixel_split_line_valid_mask]
    flat_line_coefficients = pixel_split_line_coefficients[pixel_split_line_valid_mask]
    flat_line_norm = torch.linalg.norm(
        flat_line_coefficients[:, :2],
        dim=1,
        keepdim=True,
    )
    assert torch.all(flat_line_norm.squeeze(1) > 1.0e-12), (
        "Expected every valid split line to have a non-zero direction norm. "
        f"{flat_line_norm=}"
    )
    canonical_line_coefficients = flat_line_coefficients / flat_line_norm
    flip_mask = (canonical_line_coefficients[:, 0] < 0.0) | (
        (canonical_line_coefficients[:, 0] == 0.0)
        & (canonical_line_coefficients[:, 1] < 0.0)
    )
    canonical_line_coefficients = torch.where(
        flip_mask.reshape(-1, 1),
        -canonical_line_coefficients,
        canonical_line_coefficients,
    )
    pixel_scoped_line_rows = torch.cat(
        [
            flat_pixel_indices.to(dtype=canonical_line_coefficients.dtype).reshape(
                -1,
                1,
            ),
            canonical_line_coefficients,
        ],
        dim=1,
    )
    unique_pixel_scoped_line_rows = torch.unique(
        pixel_scoped_line_rows,
        dim=0,
    )
    unique_pixel_indices = unique_pixel_scoped_line_rows[:, 0].to(dtype=torch.long)
    unique_line_coefficients = unique_pixel_scoped_line_rows[:, 1:]
    _unique_pixel_indices, pixel_group_counts = torch.unique_consecutive(
        unique_pixel_indices,
        return_counts=True,
    )
    group_start_offsets = torch.cumsum(pixel_group_counts, dim=0) - pixel_group_counts
    within_group_indices = torch.arange(
        unique_pixel_scoped_line_rows.shape[0],
        device=pixel_split_line_coefficients.device,
        dtype=torch.long,
    ) - torch.repeat_interleave(group_start_offsets, pixel_group_counts)
    deduplicated_split_line_coefficients = torch.zeros_like(
        pixel_split_line_coefficients
    )
    deduplicated_split_line_valid_mask = torch.zeros_like(pixel_split_line_valid_mask)
    deduplicated_split_line_coefficients[
        unique_pixel_indices,
        within_group_indices,
    ] = unique_line_coefficients
    deduplicated_split_line_valid_mask[
        unique_pixel_indices,
        within_group_indices,
    ] = True
    return (
        deduplicated_split_line_coefficients.contiguous(),
        deduplicated_split_line_valid_mask.contiguous(),
    )


def _build_padded_pixel_split_line_coefficients(
    pixel_indices: torch.Tensor,
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build padded polygon-edge split-line tensors for all pixels.

    Args:
        pixel_indices: Pixel indices [Np, 2] in `(y, x)` order.
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major vertex counts [Np, M].
        pixel_face_valid_mask: Pixel-major face validity mask [Np, M].

    Returns:
        Tuple of:
            split-line coefficients [Np, Lmax, 3],
            split-line validity mask [Np, Lmax].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(pixel_indices, torch.Tensor), (
            "Expected `pixel_indices` to be a tensor. " f"Got {type(pixel_indices)=}."
        )
        assert isinstance(pixel_polygon_vertices, torch.Tensor), (
            "Expected `pixel_polygon_vertices` to be a tensor. "
            f"Got {type(pixel_polygon_vertices)=}."
        )
        assert isinstance(pixel_polygon_vertex_counts, torch.Tensor), (
            "Expected `pixel_polygon_vertex_counts` to be a tensor. "
            f"Got {type(pixel_polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_face_valid_mask, torch.Tensor), (
            "Expected `pixel_face_valid_mask` to be a tensor. "
            f"Got {type(pixel_face_valid_mask)=}."
        )
        assert pixel_indices.ndim == 2, (
            "Expected `pixel_indices` to have shape `[Np, 2]`. "
            f"Got {pixel_indices.shape=}."
        )
        assert pixel_indices.shape[1] == 2, (
            "Expected `pixel_indices` to have shape `[Np, 2]`. "
            f"Got {pixel_indices.shape=}."
        )
        assert pixel_polygon_vertices.ndim == 4, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertices.shape[3] == 2, (
            "Expected `pixel_polygon_vertices` to have shape `[Np, M, Vmax, 2]`. "
            f"Got {pixel_polygon_vertices.shape=}."
        )
        assert pixel_polygon_vertex_counts.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_polygon_vertex_counts` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_polygon_vertex_counts.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_face_valid_mask.shape == pixel_polygon_vertices.shape[:2], (
            "Expected `pixel_face_valid_mask` to match the first two dimensions "
            "of `pixel_polygon_vertices`. "
            f"Got {pixel_face_valid_mask.shape=} {pixel_polygon_vertices.shape=}."
        )
        assert pixel_indices.shape[0] == pixel_polygon_vertices.shape[0], (
            "Expected one pixel index per pixel-major polygon batch item. "
            f"Got {pixel_indices.shape=} {pixel_polygon_vertices.shape=}."
        )

    _validate_inputs()

    pixel_count = pixel_polygon_vertices.shape[0]
    max_vertices_per_polygon = pixel_polygon_vertices.shape[2]
    polygon_indices = torch.arange(
        pixel_count,
        device=pixel_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(-1, 1, 1)
    face_indices = torch.arange(
        pixel_polygon_vertices.shape[1],
        device=pixel_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1, 1)
    edge_indices = torch.arange(
        max_vertices_per_polygon,
        device=pixel_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, 1, -1)
    edge_valid_mask = pixel_face_valid_mask.unsqueeze(-1) & (
        edge_indices < pixel_polygon_vertex_counts.unsqueeze(-1)
    )
    next_edge_indices = torch.where(
        edge_indices + 1 < pixel_polygon_vertex_counts.unsqueeze(-1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    edge_start = pixel_polygon_vertices
    edge_end = pixel_polygon_vertices[
        polygon_indices,
        face_indices,
        next_edge_indices,
    ]
    pixel_x = (
        pixel_indices[:, 1]
        .to(dtype=pixel_polygon_vertices.dtype)
        .reshape(
            -1,
            1,
            1,
        )
    )
    pixel_y = (
        pixel_indices[:, 0]
        .to(dtype=pixel_polygon_vertices.dtype)
        .reshape(
            -1,
            1,
            1,
        )
    )
    edge_on_pixel_boundary_mask = (
        (
            (torch.abs(edge_start[..., 0] - (pixel_x - 0.5)) <= 1.0e-6)
            & (torch.abs(edge_end[..., 0] - (pixel_x - 0.5)) <= 1.0e-6)
        )
        | (
            (torch.abs(edge_start[..., 0] - (pixel_x + 0.5)) <= 1.0e-6)
            & (torch.abs(edge_end[..., 0] - (pixel_x + 0.5)) <= 1.0e-6)
        )
        | (
            (torch.abs(edge_start[..., 1] - (pixel_y - 0.5)) <= 1.0e-6)
            & (torch.abs(edge_end[..., 1] - (pixel_y - 0.5)) <= 1.0e-6)
        )
        | (
            (torch.abs(edge_start[..., 1] - (pixel_y + 0.5)) <= 1.0e-6)
            & (torch.abs(edge_end[..., 1] - (pixel_y + 0.5)) <= 1.0e-6)
        )
    )
    edge_direction = edge_end - edge_start
    edge_line_coefficients = torch.stack(
        [
            edge_direction[..., 1],
            -edge_direction[..., 0],
            edge_direction[..., 0] * edge_start[..., 1]
            - edge_direction[..., 1] * edge_start[..., 0],
        ],
        dim=3,
    ).to(dtype=torch.float32)
    edge_valid_mask = (
        edge_valid_mask
        & (torch.linalg.norm(edge_direction, dim=3) > 1.0e-12)
        & ~edge_on_pixel_boundary_mask
    )
    edge_line_coefficients = edge_line_coefficients.reshape(pixel_count, -1, 3)
    edge_valid_mask = edge_valid_mask.reshape(pixel_count, -1)
    return _deduplicate_padded_pixel_split_lines(
        pixel_split_line_coefficients=edge_line_coefficients,
        pixel_split_line_valid_mask=edge_valid_mask,
    )


def _build_batched_pixel_cell_polygons(
    pixel_indices: torch.Tensor,
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_valid_mask: torch.Tensor,
    pixel_split_line_coefficients: torch.Tensor,
    pixel_split_line_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact arrangement cells for all pixels in batched tensor form.

    Args:
        pixel_indices: Pixel indices [Np, 2] in `(y, x)` order.
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major vertex counts [Np, M].
        pixel_face_valid_mask: Pixel-major face validity mask [Np, M].
        pixel_split_line_coefficients: Pixel-major split lines [Np, Lmax, 3].
        pixel_split_line_valid_mask: Pixel-major split-line validity mask [Np, Lmax].

    Returns:
        Tuple of:
            cell polygons [C, Vcmax, 2],
            cell polygon vertex counts [C],
            parent pixel indices [C].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(pixel_indices, torch.Tensor), (
            "Expected `pixel_indices` to be a tensor. " f"Got {type(pixel_indices)=}."
        )
        assert isinstance(pixel_polygon_vertices, torch.Tensor), (
            "Expected `pixel_polygon_vertices` to be a tensor. "
            f"Got {type(pixel_polygon_vertices)=}."
        )
        assert isinstance(pixel_polygon_vertex_counts, torch.Tensor), (
            "Expected `pixel_polygon_vertex_counts` to be a tensor. "
            f"Got {type(pixel_polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_face_valid_mask, torch.Tensor), (
            "Expected `pixel_face_valid_mask` to be a tensor. "
            f"Got {type(pixel_face_valid_mask)=}."
        )
        assert isinstance(pixel_split_line_coefficients, torch.Tensor), (
            "Expected `pixel_split_line_coefficients` to be a tensor. "
            f"Got {type(pixel_split_line_coefficients)=}."
        )
        assert isinstance(pixel_split_line_valid_mask, torch.Tensor), (
            "Expected `pixel_split_line_valid_mask` to be a tensor. "
            f"Got {type(pixel_split_line_valid_mask)=}."
        )

    _validate_inputs()

    pixel_count = pixel_polygon_vertices.shape[0]
    pixel_x = pixel_indices[:, 1].to(dtype=torch.float32)
    pixel_y = pixel_indices[:, 0].to(dtype=torch.float32)
    cell_polygon_vertices = torch.stack(
        [
            torch.stack([pixel_x - 0.5, pixel_y - 0.5], dim=1),
            torch.stack([pixel_x + 0.5, pixel_y - 0.5], dim=1),
            torch.stack([pixel_x + 0.5, pixel_y + 0.5], dim=1),
            torch.stack([pixel_x - 0.5, pixel_y + 0.5], dim=1),
        ],
        dim=1,
    ).contiguous()
    cell_polygon_vertex_counts = torch.full(
        (pixel_count,),
        fill_value=4,
        device=pixel_polygon_vertices.device,
        dtype=torch.long,
    )
    cell_pixel_indices = torch.arange(
        pixel_count,
        device=pixel_polygon_vertices.device,
        dtype=torch.long,
    )
    cell_valid_mask = torch.ones(
        (pixel_count,),
        device=pixel_polygon_vertices.device,
        dtype=torch.bool,
    )

    for split_line_index in range(pixel_split_line_coefficients.shape[1]):
        if not torch.any(cell_valid_mask):
            break
        current_max_cell_vertex_count = int(
            cell_polygon_vertex_counts[cell_valid_mask].max().item()
        )
        if cell_polygon_vertices.shape[1] != current_max_cell_vertex_count:
            cell_polygon_vertices = cell_polygon_vertices[
                :,
                :current_max_cell_vertex_count,
            ].contiguous()
        active_line_mask = (
            cell_valid_mask
            & pixel_split_line_valid_mask[
                cell_pixel_indices,
                split_line_index,
            ]
        )
        if not torch.any(active_line_mask):
            continue

        active_cell_indices = torch.nonzero(
            active_line_mask,
            as_tuple=False,
        ).reshape(-1)
        inactive_cell_mask = cell_valid_mask & ~active_line_mask
        active_cell_polygon_vertices = cell_polygon_vertices[active_cell_indices]
        active_cell_polygon_vertex_counts = cell_polygon_vertex_counts[
            active_cell_indices
        ]
        line_coefficients = pixel_split_line_coefficients[
            cell_pixel_indices[active_cell_indices],
            split_line_index,
        ]
        active_vertex_mask = torch.arange(
            active_cell_polygon_vertices.shape[1],
            device=active_cell_polygon_vertices.device,
            dtype=torch.long,
        ).reshape(1, -1) < active_cell_polygon_vertex_counts.reshape(-1, 1)
        active_line_values = (
            line_coefficients[:, 0].unsqueeze(1) * active_cell_polygon_vertices[:, :, 0]
            + line_coefficients[:, 1].unsqueeze(1)
            * active_cell_polygon_vertices[:, :, 1]
            + line_coefficients[:, 2].unsqueeze(1)
        )
        has_positive_vertex = torch.any(
            active_vertex_mask & (active_line_values > 1.0e-12),
            dim=1,
        )
        has_negative_vertex = torch.any(
            active_vertex_mask & (active_line_values < -1.0e-12),
            dim=1,
        )
        candidate_split_mask = has_positive_vertex & has_negative_vertex
        keep_active_cell_mask = ~candidate_split_mask
        if not torch.any(candidate_split_mask):
            continue

        padded_cell_polygon_vertices = F.pad(
            cell_polygon_vertices,
            pad=(0, 0, 0, 2),
        )
        active_padded_cell_polygon_vertices = padded_cell_polygon_vertices[
            active_cell_indices
        ]
        next_cell_polygon_vertices = [padded_cell_polygon_vertices[inactive_cell_mask]]
        next_cell_polygon_vertex_counts = [
            cell_polygon_vertex_counts[inactive_cell_mask]
        ]
        next_cell_pixel_indices = [cell_pixel_indices[inactive_cell_mask]]
        next_cell_valid_mask = [cell_valid_mask[inactive_cell_mask]]
        next_cell_polygon_vertices.append(
            active_padded_cell_polygon_vertices[keep_active_cell_mask]
        )
        next_cell_polygon_vertex_counts.append(
            active_cell_polygon_vertex_counts[keep_active_cell_mask]
        )
        next_cell_pixel_indices.append(
            cell_pixel_indices[active_cell_indices][keep_active_cell_mask]
        )
        next_cell_valid_mask.append(
            torch.ones_like(
                active_cell_polygon_vertex_counts[keep_active_cell_mask],
                dtype=torch.bool,
            )
        )
        candidate_cell_polygon_vertices = active_cell_polygon_vertices[
            candidate_split_mask
        ]
        candidate_padded_cell_polygon_vertices = active_padded_cell_polygon_vertices[
            candidate_split_mask
        ]
        candidate_cell_polygon_vertex_counts = active_cell_polygon_vertex_counts[
            candidate_split_mask
        ]
        candidate_cell_pixel_indices = cell_pixel_indices[active_cell_indices][
            candidate_split_mask
        ]
        candidate_line_coefficients = line_coefficients[candidate_split_mask]
        positive_polygon_vertices, positive_polygon_vertex_counts = (
            _clip_convex_polygons_to_half_plane(
                polygon_vertices=candidate_padded_cell_polygon_vertices,
                polygon_vertex_counts=candidate_cell_polygon_vertex_counts,
                line_coefficients=candidate_line_coefficients,
            )
        )
        negative_polygon_vertices, negative_polygon_vertex_counts = (
            _clip_convex_polygons_to_half_plane(
                polygon_vertices=candidate_padded_cell_polygon_vertices,
                polygon_vertex_counts=candidate_cell_polygon_vertex_counts,
                line_coefficients=-candidate_line_coefficients,
            )
        )
        positive_polygon_area = _compute_convex_polygon_areas(
            polygon_vertices=positive_polygon_vertices,
            polygon_vertex_counts=positive_polygon_vertex_counts,
        )
        negative_polygon_area = _compute_convex_polygon_areas(
            polygon_vertices=negative_polygon_vertices,
            polygon_vertex_counts=negative_polygon_vertex_counts,
        )
        split_cell_mask = (
            (positive_polygon_vertex_counts >= 3)
            & (negative_polygon_vertex_counts >= 3)
            & (positive_polygon_area > 1.0e-12)
            & (negative_polygon_area > 1.0e-12)
        )
        next_cell_polygon_vertices.append(
            candidate_padded_cell_polygon_vertices[~split_cell_mask]
        )
        next_cell_polygon_vertex_counts.append(
            candidate_cell_polygon_vertex_counts[~split_cell_mask]
        )
        next_cell_pixel_indices.append(candidate_cell_pixel_indices[~split_cell_mask])
        next_cell_valid_mask.append(
            torch.ones_like(
                candidate_cell_polygon_vertex_counts[~split_cell_mask],
                dtype=torch.bool,
            )
        )
        if torch.any(split_cell_mask):
            next_cell_polygon_vertices.extend(
                [
                    positive_polygon_vertices[split_cell_mask].contiguous(),
                    negative_polygon_vertices[split_cell_mask].contiguous(),
                ]
            )
            next_cell_polygon_vertex_counts.extend(
                [
                    positive_polygon_vertex_counts[split_cell_mask].contiguous(),
                    negative_polygon_vertex_counts[split_cell_mask].contiguous(),
                ]
            )
            next_cell_pixel_indices.extend(
                [
                    candidate_cell_pixel_indices[split_cell_mask].contiguous(),
                    candidate_cell_pixel_indices[split_cell_mask].contiguous(),
                ]
            )
            next_cell_valid_mask.extend(
                [
                    torch.ones_like(
                        positive_polygon_vertex_counts[split_cell_mask],
                        dtype=torch.bool,
                    ),
                    torch.ones_like(
                        negative_polygon_vertex_counts[split_cell_mask],
                        dtype=torch.bool,
                    ),
                ]
            )

        cell_polygon_vertices = torch.cat(
            next_cell_polygon_vertices, dim=0
        ).contiguous()
        cell_polygon_vertex_counts = torch.cat(
            next_cell_polygon_vertex_counts,
            dim=0,
        ).contiguous()
        cell_pixel_indices = torch.cat(next_cell_pixel_indices, dim=0).contiguous()
        cell_valid_mask = torch.cat(next_cell_valid_mask, dim=0).contiguous()

    final_cell_polygon_area = _compute_convex_polygon_areas(
        polygon_vertices=cell_polygon_vertices,
        polygon_vertex_counts=cell_polygon_vertex_counts,
    )
    final_cell_valid_mask = (
        cell_valid_mask
        & (cell_polygon_vertex_counts >= 3)
        & (final_cell_polygon_area > 1.0e-12)
    )
    return (
        cell_polygon_vertices[final_cell_valid_mask].contiguous(),
        cell_polygon_vertex_counts[final_cell_valid_mask].contiguous(),
        cell_pixel_indices[final_cell_valid_mask].contiguous(),
    )


def _assign_visible_faces_to_cells(
    cell_polygon_vertices: torch.Tensor,
    cell_polygon_vertex_counts: torch.Tensor,
    cell_pixel_indices: torch.Tensor,
    pixel_polygon_vertices: torch.Tensor,
    pixel_polygon_vertex_counts: torch.Tensor,
    pixel_face_indices: torch.Tensor,
    pixel_face_valid_mask: torch.Tensor,
    pixel_inverse_depth_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each batched arrangement cell to its frontmost covering face.

    Args:
        cell_polygon_vertices: Cell polygons [C, Vcmax, 2].
        cell_polygon_vertex_counts: Cell polygon vertex counts [C].
        cell_pixel_indices: Parent pixel index for each cell [C].
        pixel_polygon_vertices: Pixel-major face polygons [Np, M, Vmax, 2].
        pixel_polygon_vertex_counts: Pixel-major face vertex counts [Np, M].
        pixel_face_indices: Pixel-major local face indices [Np, M].
        pixel_face_valid_mask: Pixel-major face validity mask [Np, M].
        pixel_inverse_depth_coefficients: Pixel-major inverse-depth coefficients [Np, M, 3].

    Returns:
        Tuple of:
            visible cell polygons [Cv, Vcmax, 2],
            visible cell polygon vertex counts [Cv],
            visible local face indices [Cv].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(cell_polygon_vertices, torch.Tensor), (
            "Expected `cell_polygon_vertices` to be a tensor. "
            f"Got {type(cell_polygon_vertices)=}."
        )
        assert isinstance(cell_polygon_vertex_counts, torch.Tensor), (
            "Expected `cell_polygon_vertex_counts` to be a tensor. "
            f"Got {type(cell_polygon_vertex_counts)=}."
        )
        assert isinstance(cell_pixel_indices, torch.Tensor), (
            "Expected `cell_pixel_indices` to be a tensor. "
            f"Got {type(cell_pixel_indices)=}."
        )
        if cell_polygon_vertices.shape[0] == 0:
            return (
                cell_polygon_vertices.contiguous(),
                cell_polygon_vertex_counts.contiguous(),
                torch.zeros(
                    (0,),
                    device=cell_polygon_vertices.device,
                    dtype=torch.long,
                ),
            )

    _validate_inputs()

    cell_vertex_valid_mask = torch.arange(
        cell_polygon_vertices.shape[1],
        device=cell_polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1) < cell_polygon_vertex_counts.reshape(-1, 1)
    cell_centroid = (
        cell_polygon_vertices
        * cell_vertex_valid_mask.unsqueeze(-1).to(dtype=cell_polygon_vertices.dtype)
    ).sum(dim=1) / cell_polygon_vertex_counts.to(
        dtype=cell_polygon_vertices.dtype
    ).unsqueeze(
        1
    )

    candidate_polygon_vertices = pixel_polygon_vertices[cell_pixel_indices]
    candidate_polygon_vertex_counts = pixel_polygon_vertex_counts[cell_pixel_indices]
    candidate_face_valid_mask = pixel_face_valid_mask[cell_pixel_indices]
    candidate_inverse_depth_coefficients = pixel_inverse_depth_coefficients[
        cell_pixel_indices
    ]
    flattened_containing_face_mask = _compute_points_in_convex_polygons(
        points=cell_centroid.unsqueeze(1)
        .expand(
            -1,
            candidate_polygon_vertices.shape[1],
            -1,
        )
        .reshape(-1, 2),
        polygon_vertices=candidate_polygon_vertices.reshape(
            -1,
            candidate_polygon_vertices.shape[2],
            2,
        ),
        polygon_vertex_counts=candidate_polygon_vertex_counts.reshape(-1),
    )
    containing_face_mask = flattened_containing_face_mask.reshape(
        candidate_polygon_vertices.shape[0],
        candidate_polygon_vertices.shape[1],
    )
    depth_query = torch.stack(
        [
            cell_centroid[:, 0],
            cell_centroid[:, 1],
            torch.ones(
                (cell_centroid.shape[0],),
                device=cell_centroid.device,
                dtype=cell_centroid.dtype,
            ),
        ],
        dim=1,
    )
    candidate_inverse_depth = torch.sum(
        candidate_inverse_depth_coefficients * depth_query.unsqueeze(1),
        dim=2,
    )
    visible_face_candidate_mask = candidate_face_valid_mask & containing_face_mask
    negative_infinity = torch.full_like(
        candidate_inverse_depth,
        fill_value=-torch.inf,
    )
    candidate_inverse_depth = torch.where(
        visible_face_candidate_mask,
        candidate_inverse_depth,
        negative_infinity,
    )
    has_visible_face = torch.any(visible_face_candidate_mask, dim=1)
    visible_face_local_indices = torch.argmax(candidate_inverse_depth, dim=1)
    visible_face_indices = pixel_face_indices[
        cell_pixel_indices,
        visible_face_local_indices,
    ]
    return (
        cell_polygon_vertices[has_visible_face].contiguous(),
        cell_polygon_vertex_counts[has_visible_face].contiguous(),
        visible_face_indices[has_visible_face].contiguous(),
    )


def _pack_face_pixel_polygons_by_pixel(
    clipped_polygon_vertices: torch.Tensor,
    clipped_polygon_vertex_counts: torch.Tensor,
    clipped_pixel_indices: torch.Tensor,
    clipped_face_indices: torch.Tensor,
    face_inverse_depth_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack variable-count face-pixel polygons into pixel-major padded tensors.

    Args:
        clipped_polygon_vertices: Face-pixel polygons [P, Vmax, 2].
        clipped_polygon_vertex_counts: Valid polygon vertex counts [P].
        clipped_pixel_indices: Pixel indices [P, 2] in `(y, x)` order.
        clipped_face_indices: Local face indices [P].
        face_inverse_depth_coefficients: Inverse-depth coefficients [F, 3].

    Returns:
        Tuple of:
            pixel indices [Np, 2] in `(y, x)` order,
            pixel-major polygons [Np, Mmax, Vmax, 2],
            pixel-major vertex counts [Np, Mmax],
            pixel-major face indices [Np, Mmax],
            pixel-major valid mask [Np, Mmax],
            pixel-major inverse-depth coefficients [Np, Mmax, 3].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(
            clipped_polygon_vertices, torch.Tensor
        ), f"{type(clipped_polygon_vertices)=}"
        assert isinstance(
            clipped_polygon_vertex_counts, torch.Tensor
        ), f"{type(clipped_polygon_vertex_counts)=}"
        assert isinstance(
            clipped_pixel_indices, torch.Tensor
        ), f"{type(clipped_pixel_indices)=}"
        assert isinstance(
            clipped_face_indices, torch.Tensor
        ), f"{type(clipped_face_indices)=}"
        assert isinstance(
            face_inverse_depth_coefficients, torch.Tensor
        ), f"{type(face_inverse_depth_coefficients)=}"
        assert (
            clipped_polygon_vertices.shape[0] == clipped_pixel_indices.shape[0]
        ), f"{clipped_polygon_vertices.shape=} {clipped_pixel_indices.shape=}"
        assert (
            clipped_polygon_vertices.shape[0] == clipped_face_indices.shape[0]
        ), f"{clipped_polygon_vertices.shape=} {clipped_face_indices.shape=}"

    _validate_inputs()

    linear_pixel_indices = (
        clipped_pixel_indices[:, 0] * (clipped_pixel_indices[:, 1].amax() + 1)
        + clipped_pixel_indices[:, 1]
    )
    sorted_pair_indices = torch.argsort(linear_pixel_indices)
    sorted_linear_pixel_indices = linear_pixel_indices[sorted_pair_indices]
    _unique_linear_pixel_indices, pixel_group_counts = torch.unique_consecutive(
        sorted_linear_pixel_indices,
        return_counts=True,
    )
    pixel_count = pixel_group_counts.shape[0]
    max_faces_per_pixel = int(pixel_group_counts.max().item())
    group_start_offsets = torch.cumsum(pixel_group_counts, dim=0) - pixel_group_counts
    pixel_indices = clipped_pixel_indices[sorted_pair_indices[group_start_offsets]]
    group_indices = torch.repeat_interleave(
        torch.arange(
            pixel_count,
            device=clipped_polygon_vertices.device,
            dtype=torch.long,
        ),
        pixel_group_counts,
    )
    within_group_indices = torch.arange(
        clipped_polygon_vertices.shape[0],
        device=clipped_polygon_vertices.device,
        dtype=torch.long,
    ) - torch.repeat_interleave(group_start_offsets, pixel_group_counts)

    sorted_polygon_vertices = clipped_polygon_vertices[sorted_pair_indices]
    sorted_polygon_vertex_counts = clipped_polygon_vertex_counts[sorted_pair_indices]
    sorted_face_indices = clipped_face_indices[sorted_pair_indices]
    pixel_polygon_vertices = torch.zeros(
        (
            pixel_count,
            max_faces_per_pixel,
            clipped_polygon_vertices.shape[1],
            2,
        ),
        device=clipped_polygon_vertices.device,
        dtype=torch.float32,
    )
    pixel_polygon_vertex_counts = torch.zeros(
        (pixel_count, max_faces_per_pixel),
        device=clipped_polygon_vertices.device,
        dtype=torch.long,
    )
    pixel_face_indices = torch.full(
        (pixel_count, max_faces_per_pixel),
        fill_value=-1,
        device=clipped_polygon_vertices.device,
        dtype=torch.long,
    )
    pixel_polygon_vertices[group_indices, within_group_indices] = (
        sorted_polygon_vertices
    )
    pixel_polygon_vertex_counts[group_indices, within_group_indices] = (
        sorted_polygon_vertex_counts
    )
    pixel_face_indices[group_indices, within_group_indices] = sorted_face_indices
    pixel_face_valid_mask = pixel_face_indices >= 0
    pixel_inverse_depth_coefficients = torch.zeros(
        (pixel_count, max_faces_per_pixel, 3),
        device=clipped_polygon_vertices.device,
        dtype=torch.float32,
    )
    pixel_inverse_depth_coefficients[group_indices, within_group_indices] = (
        face_inverse_depth_coefficients[sorted_face_indices]
    )
    return (
        pixel_indices.contiguous(),
        pixel_polygon_vertices.contiguous(),
        pixel_polygon_vertex_counts.contiguous(),
        pixel_face_indices.contiguous(),
        pixel_face_valid_mask.contiguous(),
        pixel_inverse_depth_coefficients.contiguous(),
    )


def _compute_face_inverse_depth_coefficients(
    face_screen_vertices: torch.Tensor,
    face_vertex_depth: torch.Tensor,
) -> torch.Tensor:
    """Compute affine inverse-depth coefficients over projected face triangles.

    Args:
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
        face_vertex_depth: Camera-space vertex depths [F, 3].

    Returns:
        Affine inverse-depth coefficients [F, 3] for `a*x + b*y + c = 1 / z`.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert face_screen_vertices.ndim == 3, f"{face_screen_vertices.shape=}"
        assert face_screen_vertices.shape[1:] == (
            3,
            2,
        ), f"{face_screen_vertices.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_vertices.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_vertices.shape=}"
        assert torch.all(face_vertex_depth > 0.0), (
            "Expected positive camera depths for inverse-depth interpolation. "
            f"{face_vertex_depth.min()=}"
        )

    _validate_inputs()

    solve_matrix = torch.cat(
        [
            face_screen_vertices,
            torch.ones(
                (face_screen_vertices.shape[0], 3, 1),
                device=face_screen_vertices.device,
                dtype=face_screen_vertices.dtype,
            ),
        ],
        dim=2,
    )
    determinant = torch.linalg.det(solve_matrix)
    assert torch.all(torch.abs(determinant) > 1.0e-12), (
        "Expected non-degenerate projected triangles for inverse-depth solving. "
        f"{determinant.min()=}"
    )
    return (
        torch.linalg.solve(
            solve_matrix,
            (1.0 / face_vertex_depth).unsqueeze(-1),
        )
        .squeeze(-1)
        .contiguous()
    )


def _build_face_pixel_intersection_polygons(
    face_screen_vertices: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact face-pixel intersection polygons for all candidate pixels.

    Args:
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Tuple of:
            clipped polygons [P, 8, 2],
            clipped polygon vertex counts [P],
            clipped pixel indices [P, 2] in `(y, x)` order,
            clipped local face indices [P].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert face_screen_vertices.ndim == 3, f"{face_screen_vertices.shape=}"
        assert face_screen_vertices.shape[1:] == (
            3,
            2,
        ), f"{face_screen_vertices.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    face_x_min = face_screen_vertices[:, :, 0].amin(dim=1)
    face_x_max = face_screen_vertices[:, :, 0].amax(dim=1)
    face_y_min = face_screen_vertices[:, :, 1].amin(dim=1)
    face_y_max = face_screen_vertices[:, :, 1].amax(dim=1)
    pixel_x_start = torch.ceil(face_x_min - 0.5).to(dtype=torch.long)
    pixel_x_end = torch.floor(face_x_max + 0.5).to(dtype=torch.long)
    pixel_y_start = torch.ceil(face_y_min - 0.5).to(dtype=torch.long)
    pixel_y_end = torch.floor(face_y_max + 0.5).to(dtype=torch.long)
    pixel_x_start = pixel_x_start.clamp(min=0, max=image_width - 1)
    pixel_x_end = pixel_x_end.clamp(min=0, max=image_width - 1)
    pixel_y_start = pixel_y_start.clamp(min=0, max=image_height - 1)
    pixel_y_end = pixel_y_end.clamp(min=0, max=image_height - 1)
    pixel_x_count = (pixel_x_end - pixel_x_start + 1).clamp(min=0)
    pixel_y_count = (pixel_y_end - pixel_y_start + 1).clamp(min=0)
    pair_count_per_face = pixel_x_count * pixel_y_count
    total_pair_count = int(pair_count_per_face.sum().item())
    if total_pair_count == 0:
        empty_long = torch.zeros(
            (0,),
            device=face_screen_vertices.device,
            dtype=torch.long,
        )
        return (
            torch.zeros(
                (0, 8, 2),
                device=face_screen_vertices.device,
                dtype=torch.float32,
            ),
            empty_long,
            torch.zeros(
                (0, 2),
                device=face_screen_vertices.device,
                dtype=torch.long,
            ),
            empty_long,
        )

    local_face_indices = torch.arange(
        face_screen_vertices.shape[0],
        device=face_screen_vertices.device,
        dtype=torch.long,
    )
    repeated_face_indices = torch.repeat_interleave(
        local_face_indices,
        pair_count_per_face,
    )
    pair_start_offsets = torch.cumsum(pair_count_per_face, dim=0) - pair_count_per_face
    repeated_pair_start_offsets = torch.repeat_interleave(
        pair_start_offsets,
        pair_count_per_face,
    )
    repeated_pixel_x_count = torch.repeat_interleave(
        pixel_x_count,
        pair_count_per_face,
    )
    repeated_pixel_x_start = torch.repeat_interleave(
        pixel_x_start,
        pair_count_per_face,
    )
    repeated_pixel_y_start = torch.repeat_interleave(
        pixel_y_start,
        pair_count_per_face,
    )
    pair_offsets = (
        torch.arange(
            total_pair_count,
            device=face_screen_vertices.device,
            dtype=torch.long,
        )
        - repeated_pair_start_offsets
    )
    local_pixel_y_offset = torch.div(
        pair_offsets,
        repeated_pixel_x_count,
        rounding_mode="floor",
    )
    local_pixel_x_offset = pair_offsets % repeated_pixel_x_count
    pixel_x = repeated_pixel_x_start + local_pixel_x_offset
    pixel_y = repeated_pixel_y_start + local_pixel_y_offset

    polygon_vertices = torch.zeros(
        (total_pair_count, 8, 2),
        device=face_screen_vertices.device,
        dtype=torch.float32,
    )
    polygon_vertices[:, :3, :] = face_screen_vertices[repeated_face_indices].to(
        dtype=torch.float32
    )
    polygon_vertex_counts = torch.full(
        (total_pair_count,),
        fill_value=3,
        device=face_screen_vertices.device,
        dtype=torch.long,
    )
    clipped_polygon_vertices, clipped_polygon_vertex_counts = (
        _clip_convex_polygons_to_pixel_squares(
            polygon_vertices=polygon_vertices,
            polygon_vertex_counts=polygon_vertex_counts,
            pixel_x=pixel_x.to(dtype=torch.float32),
            pixel_y=pixel_y.to(dtype=torch.float32),
        )
    )
    clipped_polygon_area = _compute_convex_polygon_areas(
        polygon_vertices=clipped_polygon_vertices,
        polygon_vertex_counts=clipped_polygon_vertex_counts,
    )
    polygon_valid_mask = (clipped_polygon_vertex_counts >= 3) & (
        clipped_polygon_area > 1.0e-12
    )
    return (
        clipped_polygon_vertices[polygon_valid_mask].contiguous(),
        clipped_polygon_vertex_counts[polygon_valid_mask].contiguous(),
        torch.stack(
            [pixel_y[polygon_valid_mask], pixel_x[polygon_valid_mask]],
            dim=1,
        ).contiguous(),
        repeated_face_indices[polygon_valid_mask].contiguous(),
    )


def _compute_points_in_convex_polygons(
    points: torch.Tensor,
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
) -> torch.Tensor:
    """Test whether each point lies inside its corresponding convex polygon.

    Args:
        points: Query points [N, 2].
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid polygon vertex counts [N].

    Returns:
        Boolean containment mask [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(points, torch.Tensor), f"{type(points)=}"
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            polygon_vertex_counts, torch.Tensor
        ), f"{type(polygon_vertex_counts)=}"
        assert points.ndim == 2, f"{points.shape=}"
        assert points.shape[1] == 2, f"{points.shape=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert polygon_vertex_counts.shape == (
            polygon_vertices.shape[0],
        ), f"{polygon_vertex_counts.shape=} {polygon_vertices.shape=}"
        assert points.shape[0] == polygon_vertices.shape[0], (
            "Expected one query point per polygon. "
            f"Got {points.shape=} {polygon_vertices.shape=}."
        )

    _validate_inputs()

    polygon_count = polygon_vertices.shape[0]
    max_vertices = polygon_vertices.shape[1]
    batch_indices = torch.arange(
        polygon_count,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).unsqueeze(1)
    edge_indices = torch.arange(
        max_vertices,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).unsqueeze(0)
    edge_active = edge_indices < polygon_vertex_counts.unsqueeze(1)
    next_indices = torch.where(
        edge_indices + 1 < polygon_vertex_counts.unsqueeze(1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    current_vertices = polygon_vertices
    next_vertices = polygon_vertices[batch_indices, next_indices]
    edge_cross_values = _cross_2d(
        a=(next_vertices - current_vertices),
        b=(points.reshape(-1, 1, 2) - current_vertices),
    ).squeeze(-1)
    edge_cross_values = torch.where(
        edge_active,
        edge_cross_values,
        torch.zeros_like(edge_cross_values),
    )
    return (
        torch.all(edge_cross_values >= -1.0e-8, dim=1)
        | torch.all(edge_cross_values <= 1.0e-8, dim=1)
    ).contiguous()


def _compute_convex_polygon_bounds(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute axis-aligned bounds for convex polygons.

    Args:
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid polygon vertex counts [N].

    Returns:
        Tuple of `(x_min, x_max, y_min, y_max)` tensors [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(polygon_vertices, torch.Tensor), (
            "Expected `polygon_vertices` to be a tensor. "
            f"Got {type(polygon_vertices)=}."
        )
        assert isinstance(polygon_vertex_counts, torch.Tensor), (
            "Expected `polygon_vertex_counts` to be a tensor. "
            f"Got {type(polygon_vertex_counts)=}."
        )
        assert polygon_vertices.ndim == 3, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertices.shape[2] == 2, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertex_counts.shape == (polygon_vertices.shape[0],), (
            "Expected one polygon vertex count per polygon. "
            f"Got {polygon_vertex_counts.shape=} {polygon_vertices.shape=}."
        )

    _validate_inputs()

    if polygon_vertices.shape[0] == 0:
        empty = torch.zeros(
            (0,),
            device=polygon_vertices.device,
            dtype=polygon_vertices.dtype,
        )
        return empty, empty, empty, empty

    max_vertices = polygon_vertices.shape[1]
    vertex_indices = torch.arange(
        max_vertices,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1)
    vertex_active_mask = vertex_indices < polygon_vertex_counts.reshape(-1, 1)
    x_vertices = polygon_vertices[:, :, 0]
    y_vertices = polygon_vertices[:, :, 1]
    polygon_x_min = torch.where(
        vertex_active_mask,
        x_vertices,
        torch.full_like(x_vertices, fill_value=float("inf")),
    ).amin(dim=1)
    polygon_x_max = torch.where(
        vertex_active_mask,
        x_vertices,
        torch.full_like(x_vertices, fill_value=-float("inf")),
    ).amax(dim=1)
    polygon_y_min = torch.where(
        vertex_active_mask,
        y_vertices,
        torch.full_like(y_vertices, fill_value=float("inf")),
    ).amin(dim=1)
    polygon_y_max = torch.where(
        vertex_active_mask,
        y_vertices,
        torch.full_like(y_vertices, fill_value=-float("inf")),
    ).amax(dim=1)
    return (
        polygon_x_min.contiguous(),
        polygon_x_max.contiguous(),
        polygon_y_min.contiguous(),
        polygon_y_max.contiguous(),
    )


def _compute_points_near_convex_polygon_boundaries(
    points: torch.Tensor,
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
    squared_distance_threshold: float,
) -> torch.Tensor:
    """Test whether each point lies near its corresponding convex polygon boundary.

    Args:
        points: Query points [N, 2].
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid polygon vertex counts [N].
        squared_distance_threshold: Maximum squared Euclidean distance.

    Returns:
        Boolean near-boundary mask [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(points, torch.Tensor), (
            "Expected `points` to be a tensor. " f"Got {type(points)=}."
        )
        assert isinstance(polygon_vertices, torch.Tensor), (
            "Expected `polygon_vertices` to be a tensor. "
            f"Got {type(polygon_vertices)=}."
        )
        assert isinstance(polygon_vertex_counts, torch.Tensor), (
            "Expected `polygon_vertex_counts` to be a tensor. "
            f"Got {type(polygon_vertex_counts)=}."
        )
        assert isinstance(squared_distance_threshold, float), (
            "Expected `squared_distance_threshold` to be a float. "
            f"Got {type(squared_distance_threshold)=}."
        )
        assert points.ndim == 2, (
            "Expected `points` to have shape `[N, 2]`. " f"Got {points.shape=}."
        )
        assert points.shape[1] == 2, (
            "Expected `points` to have shape `[N, 2]`. " f"Got {points.shape=}."
        )
        assert polygon_vertices.ndim == 3, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertices.shape[2] == 2, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertex_counts.shape == (polygon_vertices.shape[0],), (
            "Expected one polygon vertex count per polygon. "
            f"Got {polygon_vertex_counts.shape=} {polygon_vertices.shape=}."
        )
        assert points.shape[0] == polygon_vertices.shape[0], (
            "Expected one query point per polygon. "
            f"Got {points.shape=} {polygon_vertices.shape=}."
        )

    _validate_inputs()

    if points.shape[0] == 0:
        return torch.zeros((0,), device=points.device, dtype=torch.bool)

    polygon_count = polygon_vertices.shape[0]
    max_vertices = polygon_vertices.shape[1]
    batch_indices = torch.arange(
        polygon_count,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    edge_indices = torch.arange(
        max_vertices,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1)
    edge_active_mask = edge_indices < polygon_vertex_counts.reshape(-1, 1)
    next_indices = torch.where(
        edge_indices + 1 < polygon_vertex_counts.reshape(-1, 1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    current_vertices = polygon_vertices
    next_vertices = polygon_vertices[batch_indices, next_indices]
    edge_vectors = next_vertices - current_vertices
    point_offsets = points.reshape(-1, 1, 2) - current_vertices
    edge_length_sq = torch.sum(edge_vectors * edge_vectors, dim=2)
    edge_projection = torch.sum(point_offsets * edge_vectors, dim=2)
    projection_t = torch.where(
        edge_length_sq > 1.0e-20,
        edge_projection / edge_length_sq,
        torch.zeros_like(edge_projection),
    ).clamp(min=0.0, max=1.0)
    closest_points = current_vertices + projection_t.unsqueeze(-1) * edge_vectors
    squared_distance = torch.sum(
        (points.reshape(-1, 1, 2) - closest_points) ** 2,
        dim=2,
    )
    squared_distance = torch.where(
        edge_active_mask,
        squared_distance,
        torch.full_like(squared_distance, fill_value=float("inf")),
    )
    return (squared_distance.amin(dim=1) <= squared_distance_threshold).contiguous()


def _compute_convex_polygon_areas(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
) -> torch.Tensor:
    """Compute areas of convex polygons.

    Args:
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid polygon vertex counts [N].

    Returns:
        Polygon areas [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            polygon_vertex_counts, torch.Tensor
        ), f"{type(polygon_vertex_counts)=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert polygon_vertex_counts.shape == (
            polygon_vertices.shape[0],
        ), f"{polygon_vertex_counts.shape=} {polygon_vertices.shape=}"

    _validate_inputs()

    polygon_count = polygon_vertices.shape[0]
    max_vertices = polygon_vertices.shape[1]
    batch_indices = torch.arange(
        polygon_count,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).unsqueeze(1)
    edge_indices = torch.arange(
        max_vertices,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).unsqueeze(0)
    edge_active = edge_indices < polygon_vertex_counts.unsqueeze(1)
    next_indices = torch.where(
        edge_indices + 1 < polygon_vertex_counts.unsqueeze(1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    current_vertices = polygon_vertices
    next_vertices = polygon_vertices[batch_indices, next_indices]
    edge_term = (
        current_vertices[:, :, 0] * next_vertices[:, :, 1]
        - current_vertices[:, :, 1] * next_vertices[:, :, 0]
    )
    double_area = torch.where(
        edge_active,
        edge_term,
        torch.zeros_like(edge_term),
    ).sum(dim=1)
    return (0.5 * torch.abs(double_area)).contiguous()


def _camera_vertices_to_pixel(
    vertices_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project camera-space vertices into image pixel coordinates.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        intrinsics: Camera intrinsics [3, 3].

    Returns:
        Pixel coordinates [V, 2] in top-first image convention.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
        assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert vertices_camera.device == intrinsics.device, (
            "Expected `vertices_camera` and `intrinsics` to share a device. "
            f"Got {vertices_camera.device=} {intrinsics.device=}."
        )

    _validate_inputs()

    vertex_x_camera = vertices_camera[:, 0]
    vertex_y_camera = vertices_camera[:, 1]
    vertex_z_camera = vertices_camera[:, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    vertex_x_pixel = fx * (vertex_x_camera / vertex_z_camera) + cx
    vertex_y_pixel = fy * (vertex_y_camera / vertex_z_camera) + cy
    return torch.stack([vertex_x_pixel, vertex_y_pixel], dim=1).contiguous()


def _clip_convex_polygons_to_pixel_squares(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip convex polygons against their corresponding pixel squares.

    Args:
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid vertex count for each polygon [N].
        pixel_x: Pixel-center x coordinate for each polygon [N].
        pixel_y: Pixel-center y coordinate for each polygon [N].

    Returns:
        Tuple of:
            clipped polygons [N, Vmax, 2],
            clipped vertex counts [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            polygon_vertex_counts, torch.Tensor
        ), f"{type(polygon_vertex_counts)=}"
        assert isinstance(pixel_x, torch.Tensor), f"{type(pixel_x)=}"
        assert isinstance(pixel_y, torch.Tensor), f"{type(pixel_y)=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert polygon_vertex_counts.ndim == 1, f"{polygon_vertex_counts.shape=}"
        assert pixel_x.ndim == 1, f"{pixel_x.shape=}"
        assert pixel_y.ndim == 1, f"{pixel_y.shape=}"
        assert (
            polygon_vertices.shape[0] == polygon_vertex_counts.shape[0]
        ), f"{polygon_vertices.shape=} {polygon_vertex_counts.shape=}"
        assert (
            pixel_x.shape[0] == polygon_vertices.shape[0]
        ), f"{pixel_x.shape=} {polygon_vertices.shape=}"
        assert (
            pixel_y.shape[0] == polygon_vertices.shape[0]
        ), f"{pixel_y.shape=} {polygon_vertices.shape=}"

    _validate_inputs()

    if torch.all(polygon_vertex_counts == 3):
        return _clip_triangle_polygons_to_pixel_squares(
            triangle_vertices=polygon_vertices[:, :3, :].contiguous(),
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            output_vertex_capacity=polygon_vertices.shape[1],
        )

    xmin = pixel_x - 0.5
    xmax = pixel_x + 0.5
    ymin = pixel_y - 0.5
    ymax = pixel_y + 0.5

    line_coefficients = [
        torch.stack(
            [torch.ones_like(xmin), torch.zeros_like(xmin), -xmin],
            dim=1,
        ),
        torch.stack(
            [-torch.ones_like(xmax), torch.zeros_like(xmax), xmax],
            dim=1,
        ),
        torch.stack(
            [torch.zeros_like(ymin), torch.ones_like(ymin), -ymin],
            dim=1,
        ),
        torch.stack(
            [torch.zeros_like(ymax), -torch.ones_like(ymax), ymax],
            dim=1,
        ),
    ]

    clipped_polygon_vertices = polygon_vertices
    clipped_polygon_vertex_counts = polygon_vertex_counts
    for coefficients in line_coefficients:
        clipped_polygon_vertices, clipped_polygon_vertex_counts = (
            _clip_convex_polygons_to_half_plane(
                polygon_vertices=clipped_polygon_vertices,
                polygon_vertex_counts=clipped_polygon_vertex_counts,
                line_coefficients=coefficients,
            )
        )
    return clipped_polygon_vertices, clipped_polygon_vertex_counts


def _clip_convex_polygons_to_half_plane(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
    line_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip convex polygons against one half-plane.

    Args:
        polygon_vertices: Convex polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid vertex count for each polygon [N].
        line_coefficients: Half-plane coefficients [N, 3] for `ax + by + c >= 0`.

    Returns:
        Tuple of:
            clipped polygons [N, Vmax, 2],
            clipped vertex counts [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            polygon_vertex_counts, torch.Tensor
        ), f"{type(polygon_vertex_counts)=}"
        assert isinstance(
            line_coefficients, torch.Tensor
        ), f"{type(line_coefficients)=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert polygon_vertex_counts.ndim == 1, f"{polygon_vertex_counts.shape=}"
        assert line_coefficients.ndim == 2, f"{line_coefficients.shape=}"
        assert line_coefficients.shape[1] == 3, f"{line_coefficients.shape=}"
        assert (
            polygon_vertices.shape[0] == polygon_vertex_counts.shape[0]
        ), f"{polygon_vertices.shape=} {polygon_vertex_counts.shape=}"
        assert (
            polygon_vertices.shape[0] == line_coefficients.shape[0]
        ), f"{polygon_vertices.shape=} {line_coefficients.shape=}"

    _validate_inputs()

    polygon_count = polygon_vertices.shape[0]
    max_vertices = polygon_vertices.shape[1]
    batch_indices = torch.arange(
        polygon_count,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    edge_indices = torch.arange(
        max_vertices,
        device=polygon_vertices.device,
        dtype=torch.long,
    ).reshape(1, -1)
    edge_active = edge_indices < polygon_vertex_counts.reshape(-1, 1)
    next_indices = torch.where(
        edge_indices + 1 < polygon_vertex_counts.reshape(-1, 1),
        edge_indices + 1,
        torch.zeros_like(edge_indices),
    )
    current_vertices = polygon_vertices
    next_vertices = polygon_vertices[batch_indices, next_indices]
    current_line_values = (
        line_coefficients[:, 0].reshape(-1, 1) * current_vertices[:, :, 0]
        + line_coefficients[:, 1].reshape(-1, 1) * current_vertices[:, :, 1]
        + line_coefficients[:, 2].reshape(-1, 1)
    )
    next_line_values = (
        line_coefficients[:, 0].reshape(-1, 1) * next_vertices[:, :, 0]
        + line_coefficients[:, 1].reshape(-1, 1) * next_vertices[:, :, 1]
        + line_coefficients[:, 2].reshape(-1, 1)
    )
    current_inside = edge_active & (current_line_values >= 0.0)
    next_inside = edge_active & (next_line_values >= 0.0)
    crossing_mask = edge_active & (current_inside != next_inside)
    edge_denominator = current_line_values - next_line_values
    edge_t = torch.zeros_like(edge_denominator)
    safe_crossing_mask = crossing_mask & (torch.abs(edge_denominator) > 1.0e-12)
    edge_t[safe_crossing_mask] = (
        current_line_values[safe_crossing_mask] / edge_denominator[safe_crossing_mask]
    )
    intersection_vertices = current_vertices + edge_t.unsqueeze(-1) * (
        next_vertices - current_vertices
    )

    candidate_vertices = torch.stack(
        [intersection_vertices, next_vertices],
        dim=2,
    ).reshape(polygon_count, 2 * max_vertices, 2)
    candidate_vertex_valid_mask = torch.stack(
        [crossing_mask, next_inside],
        dim=2,
    ).reshape(polygon_count, 2 * max_vertices)
    clipped_polygon_vertex_counts = candidate_vertex_valid_mask.sum(dim=1)
    assert torch.all(clipped_polygon_vertex_counts <= max_vertices), (
        "Expected clipped polygon output to fit the provided vertex capacity. "
        f"{clipped_polygon_vertex_counts.max()=} {max_vertices=}."
    )
    clipped_polygon_vertices = torch.zeros_like(polygon_vertices)
    if torch.any(candidate_vertex_valid_mask):
        candidate_output_indices = (
            torch.cumsum(
                candidate_vertex_valid_mask.to(dtype=torch.long),
                dim=1,
            )
            - 1
        )
        flattened_valid_mask = candidate_vertex_valid_mask.reshape(-1)
        flattened_batch_indices = (
            torch.arange(
                polygon_count,
                device=polygon_vertices.device,
                dtype=torch.long,
            )
            .reshape(-1, 1)
            .expand(-1, 2 * max_vertices)
            .reshape(-1)
        )
        clipped_polygon_vertices[
            flattened_batch_indices[flattened_valid_mask],
            candidate_output_indices.reshape(-1)[flattened_valid_mask],
        ] = candidate_vertices.reshape(-1, 2)[flattened_valid_mask]
    return clipped_polygon_vertices, clipped_polygon_vertex_counts.contiguous()


def _clip_triangle_polygons_to_pixel_squares(
    triangle_vertices: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    output_vertex_capacity: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip triangles against pixel squares with exact candidate-point geometry.

    Args:
        triangle_vertices: Triangle vertices [N, 3, 2].
        pixel_x: Pixel-center x coordinate for each triangle [N].
        pixel_y: Pixel-center y coordinate for each triangle [N].
        output_vertex_capacity: Output polygon capacity.

    Returns:
        Tuple of:
            clipped polygons [N, Vmax, 2],
            clipped vertex counts [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(triangle_vertices, torch.Tensor), (
            "Expected `triangle_vertices` to be a tensor. "
            f"Got {type(triangle_vertices)=}."
        )
        assert isinstance(pixel_x, torch.Tensor), (
            "Expected `pixel_x` to be a tensor. " f"Got {type(pixel_x)=}."
        )
        assert isinstance(pixel_y, torch.Tensor), (
            "Expected `pixel_y` to be a tensor. " f"Got {type(pixel_y)=}."
        )
        assert isinstance(output_vertex_capacity, int), (
            "Expected `output_vertex_capacity` to be an int. "
            f"Got {type(output_vertex_capacity)=}."
        )
        assert triangle_vertices.ndim == 3, (
            "Expected `triangle_vertices` to have shape `[N, 3, 2]`. "
            f"Got {triangle_vertices.shape=}."
        )
        assert triangle_vertices.shape[1:] == (3, 2), (
            "Expected `triangle_vertices` to have shape `[N, 3, 2]`. "
            f"Got {triangle_vertices.shape=}."
        )
        assert pixel_x.ndim == 1, f"{pixel_x.shape=}"
        assert pixel_y.ndim == 1, f"{pixel_y.shape=}"
        assert triangle_vertices.shape[0] == pixel_x.shape[0], (
            "Expected one pixel x coordinate per triangle. "
            f"Got {triangle_vertices.shape=} {pixel_x.shape=}."
        )
        assert triangle_vertices.shape[0] == pixel_y.shape[0], (
            "Expected one pixel y coordinate per triangle. "
            f"Got {triangle_vertices.shape=} {pixel_y.shape=}."
        )
        assert output_vertex_capacity >= 3, (
            "Expected `output_vertex_capacity` to be at least 3. "
            f"Got {output_vertex_capacity=}."
        )

    _validate_inputs()

    triangle_count = triangle_vertices.shape[0]
    if triangle_count == 0:
        return (
            torch.zeros(
                (0, output_vertex_capacity, 2),
                device=triangle_vertices.device,
                dtype=triangle_vertices.dtype,
            ),
            torch.zeros(
                (0,),
                device=triangle_vertices.device,
                dtype=torch.long,
            ),
        )

    xmin = pixel_x - 0.5
    xmax = pixel_x + 0.5
    ymin = pixel_y - 0.5
    ymax = pixel_y + 0.5
    eps = 1.0e-6

    square_corners = torch.stack(
        [
            torch.stack([xmin, ymin], dim=1),
            torch.stack([xmax, ymin], dim=1),
            torch.stack([xmax, ymax], dim=1),
            torch.stack([xmin, ymax], dim=1),
        ],
        dim=1,
    ).to(dtype=triangle_vertices.dtype)
    triangle_vertex_inside_mask = (
        (triangle_vertices[:, :, 0] >= xmin.unsqueeze(1) - eps)
        & (triangle_vertices[:, :, 0] <= xmax.unsqueeze(1) + eps)
        & (triangle_vertices[:, :, 1] >= ymin.unsqueeze(1) - eps)
        & (triangle_vertices[:, :, 1] <= ymax.unsqueeze(1) + eps)
    )
    square_corner_inside_mask = _compute_points_in_triangles(
        points=square_corners,
        triangle_vertices=triangle_vertices,
    )

    edge_start = triangle_vertices
    edge_end = triangle_vertices[:, [1, 2, 0], :]
    edge_direction = edge_end - edge_start
    edge_dx = edge_direction[:, :, 0]
    edge_dy = edge_direction[:, :, 1]
    vertical_boundaries = torch.stack([xmin, xmax], dim=1).to(
        dtype=triangle_vertices.dtype
    )
    horizontal_boundaries = torch.stack([ymin, ymax], dim=1).to(
        dtype=triangle_vertices.dtype
    )

    safe_edge_dx = torch.where(
        torch.abs(edge_dx) > 1.0e-12,
        edge_dx,
        torch.ones_like(edge_dx),
    )
    vertical_t = (
        vertical_boundaries.unsqueeze(1) - edge_start[:, :, 0].unsqueeze(2)
    ) / safe_edge_dx.unsqueeze(2)
    vertical_intersection_y = edge_start[:, :, 1].unsqueeze(2) + vertical_t * (
        edge_dy.unsqueeze(2)
    )
    vertical_intersection_points = torch.stack(
        [
            vertical_boundaries.unsqueeze(1).expand(-1, 3, -1),
            vertical_intersection_y,
        ],
        dim=3,
    )
    vertical_intersection_mask = (
        (torch.abs(edge_dx).unsqueeze(2) > 1.0e-12)
        & (vertical_t >= -eps)
        & (vertical_t <= 1.0 + eps)
        & (vertical_intersection_y >= ymin.reshape(-1, 1, 1) - eps)
        & (vertical_intersection_y <= ymax.reshape(-1, 1, 1) + eps)
    )

    safe_edge_dy = torch.where(
        torch.abs(edge_dy) > 1.0e-12,
        edge_dy,
        torch.ones_like(edge_dy),
    )
    horizontal_t = (
        horizontal_boundaries.unsqueeze(1) - edge_start[:, :, 1].unsqueeze(2)
    ) / safe_edge_dy.unsqueeze(2)
    horizontal_intersection_x = edge_start[:, :, 0].unsqueeze(2) + horizontal_t * (
        edge_dx.unsqueeze(2)
    )
    horizontal_intersection_points = torch.stack(
        [
            horizontal_intersection_x,
            horizontal_boundaries.unsqueeze(1).expand(-1, 3, -1),
        ],
        dim=3,
    )
    horizontal_intersection_mask = (
        (torch.abs(edge_dy).unsqueeze(2) > 1.0e-12)
        & (horizontal_t >= -eps)
        & (horizontal_t <= 1.0 + eps)
        & (horizontal_intersection_x >= xmin.reshape(-1, 1, 1) - eps)
        & (horizontal_intersection_x <= xmax.reshape(-1, 1, 1) + eps)
    )

    candidate_points = torch.cat(
        [
            triangle_vertices,
            square_corners,
            vertical_intersection_points.reshape(triangle_count, 6, 2),
            horizontal_intersection_points.reshape(triangle_count, 6, 2),
        ],
        dim=1,
    )
    candidate_mask = torch.cat(
        [
            triangle_vertex_inside_mask,
            square_corner_inside_mask,
            vertical_intersection_mask.reshape(triangle_count, 6),
            horizontal_intersection_mask.reshape(triangle_count, 6),
        ],
        dim=1,
    )
    candidate_count = candidate_points.shape[1]
    pairwise_same_mask = (
        torch.max(
            torch.abs(candidate_points.unsqueeze(2) - candidate_points.unsqueeze(1)),
            dim=3,
        ).values
        <= eps
    )
    pairwise_same_mask = (
        pairwise_same_mask & candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
    )
    earlier_duplicate_mask = torch.tril(
        torch.ones(
            (candidate_count, candidate_count),
            device=triangle_vertices.device,
            dtype=torch.bool,
        ),
        diagonal=-1,
    )
    unique_candidate_mask = candidate_mask & ~torch.any(
        pairwise_same_mask & earlier_duplicate_mask.unsqueeze(0),
        dim=2,
    )
    unique_candidate_count = unique_candidate_mask.sum(dim=1)
    assert torch.all(unique_candidate_count <= output_vertex_capacity), (
        "Expected triangle-square intersections to fit in the output capacity. "
        f"{unique_candidate_count.max()=} {output_vertex_capacity=}."
    )

    unique_candidate_points = torch.where(
        unique_candidate_mask.unsqueeze(-1),
        candidate_points,
        torch.zeros_like(candidate_points),
    )
    safe_unique_candidate_count = unique_candidate_count.clamp(min=1).to(
        dtype=triangle_vertices.dtype
    )
    polygon_centroid = unique_candidate_points.sum(
        dim=1
    ) / safe_unique_candidate_count.unsqueeze(1)
    candidate_angles = torch.atan2(
        candidate_points[:, :, 1] - polygon_centroid[:, 1].unsqueeze(1),
        candidate_points[:, :, 0] - polygon_centroid[:, 0].unsqueeze(1),
    )
    candidate_angles = torch.where(
        unique_candidate_mask,
        candidate_angles,
        torch.full_like(candidate_angles, fill_value=torch.inf),
    )
    sorted_candidate_indices = torch.argsort(candidate_angles, dim=1)
    sorted_candidate_points = torch.gather(
        candidate_points,
        dim=1,
        index=sorted_candidate_indices.unsqueeze(-1).expand(-1, -1, 2),
    )
    sorted_candidate_mask = torch.gather(
        unique_candidate_mask.to(dtype=torch.long),
        dim=1,
        index=sorted_candidate_indices,
    ).to(dtype=torch.bool)
    clipped_polygon_vertices = torch.zeros(
        (triangle_count, output_vertex_capacity, 2),
        device=triangle_vertices.device,
        dtype=triangle_vertices.dtype,
    )
    clipped_polygon_vertices[:, :candidate_count, :] = torch.where(
        sorted_candidate_mask.unsqueeze(-1),
        sorted_candidate_points,
        torch.zeros_like(sorted_candidate_points),
    )[:, :output_vertex_capacity, :]
    clipped_polygon_vertex_counts = unique_candidate_count.to(dtype=torch.long)
    return clipped_polygon_vertices.contiguous(), clipped_polygon_vertex_counts


def _project_screen_polygons_to_face_uv(
    polygon_vertices: torch.Tensor,
    face_screen_vertices: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    face_vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Map image-space polygon vertices to UV using exact perspective interpolation.

    Args:
        polygon_vertices: Image-space polygons [N, Vmax, 2].
        face_screen_vertices: Projected triangle vertices [N, 3, 2].
        face_vertex_depth: Camera-space triangle depths [N, 3].
        face_vertex_uv: Seam-safe triangle UV coordinates [N, 3, 2].

    Returns:
        UV-space polygons [N, Vmax, 2].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(face_vertex_uv, torch.Tensor), f"{type(face_vertex_uv)=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert face_screen_vertices.shape == (
            polygon_vertices.shape[0],
            3,
            2,
        ), f"{face_screen_vertices.shape=} {polygon_vertices.shape=}"
        assert face_vertex_depth.shape == (
            polygon_vertices.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {polygon_vertices.shape=}"
        assert face_vertex_uv.shape == (
            polygon_vertices.shape[0],
            3,
            2,
        ), f"{face_vertex_uv.shape=} {polygon_vertices.shape=}"

    _validate_inputs()

    face_screen_v0 = face_screen_vertices[:, 0:1, :]
    face_screen_v1 = face_screen_vertices[:, 1:2, :]
    face_screen_v2 = face_screen_vertices[:, 2:3, :]
    denominator = _cross_2d(
        a=(face_screen_v1 - face_screen_v0),
        b=(face_screen_v2 - face_screen_v0),
    ).squeeze(-1)
    assert torch.all(torch.abs(denominator) > 1.0e-12), f"{denominator=}"

    barycentric_0 = (
        _cross_2d(
            a=(face_screen_v1 - polygon_vertices),
            b=(face_screen_v2 - polygon_vertices),
        ).squeeze(-1)
        / denominator
    )
    barycentric_1 = (
        _cross_2d(
            a=(face_screen_v2 - polygon_vertices),
            b=(face_screen_v0 - polygon_vertices),
        ).squeeze(-1)
        / denominator
    )
    barycentric_2 = 1.0 - barycentric_0 - barycentric_1
    barycentric = torch.stack(
        [barycentric_0, barycentric_1, barycentric_2],
        dim=2,
    )
    inverse_depth = 1.0 / face_vertex_depth
    perspective_weight = barycentric * inverse_depth.unsqueeze(1)
    perspective_weight = perspective_weight / perspective_weight.sum(
        dim=2,
        keepdim=True,
    )
    return torch.sum(
        perspective_weight.unsqueeze(-1) * face_vertex_uv.unsqueeze(1),
        dim=2,
    ).contiguous()


def _cross_2d(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D cross product magnitude.

    Args:
        a: 2D vectors [..., 2].
        b: 2D vectors [..., 2].

    Returns:
        Scalar cross product [..., 1].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(a, torch.Tensor), f"{type(a)=}"
        assert isinstance(b, torch.Tensor), f"{type(b)=}"
        assert a.shape == b.shape, f"{a.shape=} {b.shape=}"
        assert a.shape[-1] == 2, f"{a.shape=}"

    _validate_inputs()

    return (a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]).contiguous()


def _compute_points_in_triangles(
    points: torch.Tensor,
    triangle_vertices: torch.Tensor,
) -> torch.Tensor:
    """Test whether batched points lie inside their corresponding triangles.

    Args:
        points: Query points [N, P, 2].
        triangle_vertices: Triangle vertices [N, 3, 2].

    Returns:
        Boolean containment mask [N, P].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(points, torch.Tensor), f"{type(points)=}"
        assert isinstance(
            triangle_vertices, torch.Tensor
        ), f"{type(triangle_vertices)=}"
        assert points.ndim == 3, f"{points.shape=}"
        assert points.shape[2] == 2, f"{points.shape=}"
        assert triangle_vertices.ndim == 3, f"{triangle_vertices.shape=}"
        assert triangle_vertices.shape[1:] == (3, 2), f"{triangle_vertices.shape=}"
        assert points.shape[0] == triangle_vertices.shape[0], (
            "Expected one triangle per point batch. "
            f"Got {points.shape=} {triangle_vertices.shape=}."
        )

    _validate_inputs()

    triangle_v0 = triangle_vertices[:, 0:1, :]
    triangle_v1 = triangle_vertices[:, 1:2, :]
    triangle_v2 = triangle_vertices[:, 2:3, :]
    point_count = points.shape[1]
    edge01_cross = _cross_2d(
        a=(triangle_v1 - triangle_v0).expand(-1, point_count, -1),
        b=points - triangle_v0,
    ).squeeze(-1)
    edge12_cross = _cross_2d(
        a=(triangle_v2 - triangle_v1).expand(-1, point_count, -1),
        b=points - triangle_v1,
    ).squeeze(-1)
    edge20_cross = _cross_2d(
        a=(triangle_v0 - triangle_v2).expand(-1, point_count, -1),
        b=points - triangle_v2,
    ).squeeze(-1)
    return (
        (
            (edge01_cross >= -1.0e-8)
            & (edge12_cross >= -1.0e-8)
            & (edge20_cross >= -1.0e-8)
        )
        | (
            (edge01_cross <= 1.0e-8)
            & (edge12_cross <= 1.0e-8)
            & (edge20_cross <= 1.0e-8)
        )
    ).contiguous()


def _compute_convex_polygon_pixel_square_positive_area_overlap_mask(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
) -> torch.Tensor:
    """Detect positive-area overlap between convex polygons and pixel squares.

    Args:
        polygon_vertices: Convex polygon vertices [N, Vmax, 2].
        polygon_vertex_counts: Valid polygon vertex counts [N].
        pixel_x: Pixel-center x coordinate for each polygon [N].
        pixel_y: Pixel-center y coordinate for each polygon [N].

    Returns:
        Boolean positive-area overlap mask [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(polygon_vertices, torch.Tensor), (
            "Expected `polygon_vertices` to be a tensor. "
            f"Got {type(polygon_vertices)=}."
        )
        assert isinstance(polygon_vertex_counts, torch.Tensor), (
            "Expected `polygon_vertex_counts` to be a tensor. "
            f"Got {type(polygon_vertex_counts)=}."
        )
        assert isinstance(pixel_x, torch.Tensor), (
            "Expected `pixel_x` to be a tensor. " f"Got {type(pixel_x)=}."
        )
        assert isinstance(pixel_y, torch.Tensor), (
            "Expected `pixel_y` to be a tensor. " f"Got {type(pixel_y)=}."
        )
        assert polygon_vertices.ndim == 3, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertices.shape[2] == 2, (
            "Expected `polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {polygon_vertices.shape=}."
        )
        assert polygon_vertex_counts.shape == (polygon_vertices.shape[0],), (
            "Expected one polygon vertex count per polygon. "
            f"Got {polygon_vertex_counts.shape=} {polygon_vertices.shape=}."
        )
        assert pixel_x.shape == (polygon_vertices.shape[0],), (
            "Expected one pixel x coordinate per polygon. "
            f"Got {pixel_x.shape=} {polygon_vertices.shape=}."
        )
        assert pixel_y.shape == (polygon_vertices.shape[0],), (
            "Expected one pixel y coordinate per polygon. "
            f"Got {pixel_y.shape=} {polygon_vertices.shape=}."
        )

    _validate_inputs()

    polygon_count = polygon_vertices.shape[0]
    if polygon_count == 0:
        return torch.zeros((0,), device=polygon_vertices.device, dtype=torch.bool)

    polygon_x_min, polygon_x_max, polygon_y_min, polygon_y_max = (
        _compute_convex_polygon_bounds(
            polygon_vertices=polygon_vertices,
            polygon_vertex_counts=polygon_vertex_counts,
        )
    )
    pixel_x_min = pixel_x - 0.5
    pixel_x_max = pixel_x + 0.5
    pixel_y_min = pixel_y - 0.5
    pixel_y_max = pixel_y + 0.5
    bbox_overlap_mask = (
        (polygon_x_min < pixel_x_max)
        & (pixel_x_min < polygon_x_max)
        & (polygon_y_min < pixel_y_max)
        & (pixel_y_min < polygon_y_max)
    )
    positive_area_overlap_mask = torch.zeros(
        (polygon_count,),
        device=polygon_vertices.device,
        dtype=torch.bool,
    )
    if not torch.any(bbox_overlap_mask):
        return positive_area_overlap_mask.contiguous()

    unclipped_polygon_vertices = polygon_vertices[bbox_overlap_mask]
    unclipped_polygon_vertex_capacity = unclipped_polygon_vertices.shape[1]
    clipped_polygon_input_vertices = torch.zeros(
        (
            unclipped_polygon_vertices.shape[0],
            unclipped_polygon_vertex_capacity + 4,
            2,
        ),
        device=polygon_vertices.device,
        dtype=polygon_vertices.dtype,
    )
    clipped_polygon_input_vertices[:, :unclipped_polygon_vertex_capacity, :] = (
        unclipped_polygon_vertices
    )
    (
        clipped_polygon_vertices,
        clipped_polygon_vertex_counts,
    ) = _clip_convex_polygons_to_pixel_squares(
        polygon_vertices=clipped_polygon_input_vertices,
        polygon_vertex_counts=polygon_vertex_counts[bbox_overlap_mask],
        pixel_x=pixel_x[bbox_overlap_mask],
        pixel_y=pixel_y[bbox_overlap_mask],
    )
    clipped_polygon_area = _compute_convex_polygon_areas(
        polygon_vertices=clipped_polygon_vertices,
        polygon_vertex_counts=clipped_polygon_vertex_counts,
    )
    positive_area_overlap_mask[bbox_overlap_mask] = (
        clipped_polygon_vertex_counts >= 3
    ) & (clipped_polygon_area > 1.0e-12)
    return positive_area_overlap_mask.contiguous()


def _duplicate_wrapped_uv_polygons(
    uv_polygon_vertices: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Duplicate UV polygons across the cylindrical wrap boundary when needed.

    Args:
        uv_polygon_vertices: Convex UV polygons [N, Vmax, 2].
        uv_polygon_vertex_counts: Valid polygon vertex counts [N].

    Returns:
        Tuple of:
            wrapped UV polygons [Nw, Vmax, 2],
            wrapped UV polygon vertex counts [Nw].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(uv_polygon_vertices, torch.Tensor), (
            "Expected `uv_polygon_vertices` to be a tensor. "
            f"Got {type(uv_polygon_vertices)=}."
        )
        assert isinstance(uv_polygon_vertex_counts, torch.Tensor), (
            "Expected `uv_polygon_vertex_counts` to be a tensor. "
            f"Got {type(uv_polygon_vertex_counts)=}."
        )
        assert uv_polygon_vertices.ndim == 3, (
            "Expected `uv_polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_vertices.shape=}."
        )
        assert uv_polygon_vertices.shape[2] == 2, (
            "Expected `uv_polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_vertices.shape=}."
        )
        assert uv_polygon_vertex_counts.shape == (uv_polygon_vertices.shape[0],), (
            "Expected one polygon vertex count per polygon. "
            f"Got {uv_polygon_vertex_counts.shape=} {uv_polygon_vertices.shape=}."
        )

    _validate_inputs()

    if uv_polygon_vertices.shape[0] == 0:
        return (
            uv_polygon_vertices.to(dtype=torch.float32).contiguous(),
            uv_polygon_vertex_counts.contiguous(),
        )

    polygon_x_min, polygon_x_max, _, _ = _compute_convex_polygon_bounds(
        polygon_vertices=uv_polygon_vertices,
        polygon_vertex_counts=uv_polygon_vertex_counts,
    )
    wrapped_polygon_vertex_chunks = [
        uv_polygon_vertices.to(dtype=torch.float32).contiguous()
    ]
    wrapped_polygon_count_chunks = [uv_polygon_vertex_counts.contiguous()]
    polygon_extends_right = polygon_x_max > 1.0
    polygon_extends_left = polygon_x_min < 0.0
    if torch.any(polygon_extends_right):
        shifted_right = uv_polygon_vertices[polygon_extends_right].to(
            dtype=torch.float32
        )
        shifted_right = shifted_right.clone()
        shifted_right[:, :, 0] = shifted_right[:, :, 0] - 1.0
        wrapped_polygon_vertex_chunks.append(shifted_right.contiguous())
        wrapped_polygon_count_chunks.append(
            uv_polygon_vertex_counts[polygon_extends_right].contiguous()
        )
    if torch.any(polygon_extends_left):
        shifted_left = uv_polygon_vertices[polygon_extends_left].to(dtype=torch.float32)
        shifted_left = shifted_left.clone()
        shifted_left[:, :, 0] = shifted_left[:, :, 0] + 1.0
        wrapped_polygon_vertex_chunks.append(shifted_left.contiguous())
        wrapped_polygon_count_chunks.append(
            uv_polygon_vertex_counts[polygon_extends_left].contiguous()
        )
    return (
        torch.cat(wrapped_polygon_vertex_chunks, dim=0).contiguous(),
        torch.cat(wrapped_polygon_count_chunks, dim=0).contiguous(),
    )


def _build_uv_polygon_texel_intersections(
    uv_polygon_vertices: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Build exact UV-polygon to texel-cell intersection indices.

    Args:
        uv_polygon_vertices: Convex UV polygons [N, Vmax, 2].
        uv_polygon_vertex_counts: Valid polygon vertex counts [N].
        texture_size: UV texture resolution.

    Returns:
        Covered texel indices [M, 2] in `(row, col)` order.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(uv_polygon_vertices, torch.Tensor), (
            "Expected `uv_polygon_vertices` to be a tensor. "
            f"Got {type(uv_polygon_vertices)=}."
        )
        assert isinstance(uv_polygon_vertex_counts, torch.Tensor), (
            "Expected `uv_polygon_vertex_counts` to be a tensor. "
            f"Got {type(uv_polygon_vertex_counts)=}."
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an int. " f"Got {type(texture_size)=}."
        )
        assert uv_polygon_vertices.ndim == 3, (
            "Expected `uv_polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_vertices.shape=}."
        )
        assert uv_polygon_vertices.shape[2] == 2, (
            "Expected `uv_polygon_vertices` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_vertices.shape=}."
        )
        assert uv_polygon_vertex_counts.shape == (uv_polygon_vertices.shape[0],), (
            "Expected one polygon vertex count per polygon. "
            f"Got {uv_polygon_vertex_counts.shape=} {uv_polygon_vertices.shape=}."
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"Got {texture_size=}."
        )

    _validate_inputs()

    if uv_polygon_vertices.shape[0] == 0:
        return torch.zeros(
            (0, 2),
            device=uv_polygon_vertices.device,
            dtype=torch.long,
        )

    polygon_texel_vertices = uv_polygon_vertices.to(dtype=torch.float32) * float(
        texture_size
    )
    polygon_x_min, polygon_x_max, polygon_y_min, polygon_y_max = (
        _compute_convex_polygon_bounds(
            polygon_vertices=polygon_texel_vertices,
            polygon_vertex_counts=uv_polygon_vertex_counts,
        )
    )
    texel_x_start = torch.floor(polygon_x_min).to(dtype=torch.long)
    texel_x_end = torch.ceil(polygon_x_max).to(dtype=torch.long) - 1
    texel_y_start = torch.floor(polygon_y_min).to(dtype=torch.long)
    texel_y_end = torch.ceil(polygon_y_max).to(dtype=torch.long) - 1
    texel_x_start = texel_x_start.clamp(min=0, max=texture_size - 1)
    texel_x_end = texel_x_end.clamp(min=0, max=texture_size - 1)
    texel_y_start = texel_y_start.clamp(min=0, max=texture_size - 1)
    texel_y_end = texel_y_end.clamp(min=0, max=texture_size - 1)
    texel_x_count = (texel_x_end - texel_x_start + 1).clamp(min=0)
    texel_y_count = (texel_y_end - texel_y_start + 1).clamp(min=0)
    pair_count_per_polygon = texel_x_count * texel_y_count
    total_pair_count = int(pair_count_per_polygon.sum().item())
    if total_pair_count == 0:
        return torch.zeros(
            (0, 2),
            device=uv_polygon_vertices.device,
            dtype=torch.long,
        )

    covered_texel_mask = torch.zeros(
        (texture_size, texture_size),
        device=uv_polygon_vertices.device,
        dtype=torch.bool,
    )
    target_uv_polygon_texel_pair_budget = 262144
    boundary_squared_distance_threshold = 0.501
    chunk_start = 0
    while chunk_start < uv_polygon_vertices.shape[0]:
        chunk_end = chunk_start + 1
        chunk_pair_count = int(pair_count_per_polygon[chunk_start].item())
        while chunk_end < uv_polygon_vertices.shape[0]:
            next_pair_count = int(pair_count_per_polygon[chunk_end].item())
            if chunk_pair_count + next_pair_count > target_uv_polygon_texel_pair_budget:
                break
            chunk_pair_count += next_pair_count
            chunk_end += 1

        chunk_polygon_indices = torch.arange(
            chunk_start,
            chunk_end,
            device=uv_polygon_vertices.device,
            dtype=torch.long,
        )
        chunk_pair_count_per_polygon = pair_count_per_polygon[chunk_start:chunk_end]
        repeated_polygon_indices = torch.repeat_interleave(
            chunk_polygon_indices,
            chunk_pair_count_per_polygon,
        )
        if repeated_polygon_indices.shape[0] == 0:
            chunk_start = chunk_end
            continue

        pair_start_offsets = (
            torch.cumsum(
                chunk_pair_count_per_polygon,
                dim=0,
            )
            - chunk_pair_count_per_polygon
        )
        repeated_pair_start_offsets = torch.repeat_interleave(
            pair_start_offsets,
            chunk_pair_count_per_polygon,
        )
        repeated_texel_x_count = torch.repeat_interleave(
            texel_x_count[chunk_start:chunk_end],
            chunk_pair_count_per_polygon,
        )
        repeated_texel_x_start = torch.repeat_interleave(
            texel_x_start[chunk_start:chunk_end],
            chunk_pair_count_per_polygon,
        )
        repeated_texel_y_start = torch.repeat_interleave(
            texel_y_start[chunk_start:chunk_end],
            chunk_pair_count_per_polygon,
        )
        pair_offsets = (
            torch.arange(
                chunk_pair_count,
                device=uv_polygon_vertices.device,
                dtype=torch.long,
            )
            - repeated_pair_start_offsets
        )
        texel_y = repeated_texel_y_start + torch.div(
            pair_offsets,
            repeated_texel_x_count,
            rounding_mode="floor",
        )
        texel_x = repeated_texel_x_start + (pair_offsets % repeated_texel_x_count)
        pixel_x = texel_x.to(dtype=torch.float32) + 0.5
        pixel_y = texel_y.to(dtype=torch.float32) + 0.5
        pixel_centers = torch.stack([pixel_x, pixel_y], dim=1)
        candidate_polygon_vertices = polygon_texel_vertices[repeated_polygon_indices]
        candidate_polygon_vertex_counts = uv_polygon_vertex_counts[
            repeated_polygon_indices
        ]
        interior_mask = _compute_points_in_convex_polygons(
            points=pixel_centers,
            polygon_vertices=candidate_polygon_vertices,
            polygon_vertex_counts=candidate_polygon_vertex_counts,
        )
        near_boundary_mask = _compute_points_near_convex_polygon_boundaries(
            points=pixel_centers,
            polygon_vertices=candidate_polygon_vertices,
            polygon_vertex_counts=candidate_polygon_vertex_counts,
            squared_distance_threshold=boundary_squared_distance_threshold,
        )
        boundary_candidate_mask = near_boundary_mask
        accepted_mask = interior_mask & (~near_boundary_mask)
        if torch.any(boundary_candidate_mask):
            boundary_polygon_vertices = candidate_polygon_vertices[
                boundary_candidate_mask
            ]
            boundary_polygon_vertex_counts = candidate_polygon_vertex_counts[
                boundary_candidate_mask
            ]
            boundary_pixel_x = pixel_x[boundary_candidate_mask]
            boundary_pixel_y = pixel_y[boundary_candidate_mask]
            boundary_triangle_chunks: List[torch.Tensor] = []
            boundary_triangle_candidate_index_chunks: List[torch.Tensor] = []
            for fan_index in range(1, boundary_polygon_vertices.shape[1] - 1):
                fan_valid_mask = boundary_polygon_vertex_counts > (fan_index + 1)
                if not torch.any(fan_valid_mask):
                    continue
                boundary_triangle_chunks.append(
                    torch.stack(
                        [
                            boundary_polygon_vertices[fan_valid_mask, 0, :],
                            boundary_polygon_vertices[fan_valid_mask, fan_index, :],
                            boundary_polygon_vertices[fan_valid_mask, fan_index + 1, :],
                        ],
                        dim=1,
                    )
                )
                boundary_triangle_candidate_index_chunks.append(
                    torch.nonzero(
                        fan_valid_mask,
                        as_tuple=False,
                    ).reshape(-1)
                )
            if len(boundary_triangle_chunks) > 0:
                boundary_triangles = torch.cat(
                    boundary_triangle_chunks,
                    dim=0,
                ).contiguous()
                boundary_triangle_candidate_indices = torch.cat(
                    boundary_triangle_candidate_index_chunks,
                    dim=0,
                ).contiguous()
                boundary_triangle_overlap_mask = (
                    _compute_triangle_pixel_square_positive_area_overlap_mask(
                        triangle_vertices=boundary_triangles,
                        pixel_x=boundary_pixel_x[boundary_triangle_candidate_indices],
                        pixel_y=boundary_pixel_y[boundary_triangle_candidate_indices],
                    )
                )
                boundary_candidate_positive_mask = torch.zeros(
                    (boundary_polygon_vertices.shape[0],),
                    device=boundary_polygon_vertices.device,
                    dtype=torch.bool,
                )
                boundary_candidate_positive_mask[
                    boundary_triangle_candidate_indices[boundary_triangle_overlap_mask]
                ] = True
                accepted_mask[boundary_candidate_mask] = (
                    boundary_candidate_positive_mask
                )
        if torch.any(accepted_mask):
            covered_texel_mask[
                texel_y[accepted_mask],
                texel_x[accepted_mask],
            ] = True
        chunk_start = chunk_end

    return torch.nonzero(covered_texel_mask, as_tuple=False).contiguous()


def _triangulate_convex_uv_polygons(
    polygon_vertices: torch.Tensor,
    polygon_vertex_counts: torch.Tensor,
) -> torch.Tensor:
    """Triangulate convex UV polygons into a triangle soup.

    Args:
        polygon_vertices: Convex UV polygons [N, Vmax, 2].
        polygon_vertex_counts: Valid vertex count for each polygon [N].

    Returns:
        UV triangle soup [K, 3, 2].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(polygon_vertices, torch.Tensor), f"{type(polygon_vertices)=}"
        assert isinstance(
            polygon_vertex_counts, torch.Tensor
        ), f"{type(polygon_vertex_counts)=}"
        assert polygon_vertices.ndim == 3, f"{polygon_vertices.shape=}"
        assert polygon_vertices.shape[2] == 2, f"{polygon_vertices.shape=}"
        assert polygon_vertex_counts.ndim == 1, f"{polygon_vertex_counts.shape=}"
        assert (
            polygon_vertices.shape[0] == polygon_vertex_counts.shape[0]
        ), f"{polygon_vertices.shape=} {polygon_vertex_counts.shape=}"

    _validate_inputs()

    uv_triangle_chunks: List[torch.Tensor] = []
    for fan_index in range(1, polygon_vertices.shape[1] - 1):
        fan_valid_mask = polygon_vertex_counts > (fan_index + 1)
        if not torch.any(fan_valid_mask):
            continue
        uv_triangle_chunks.append(
            torch.stack(
                [
                    polygon_vertices[fan_valid_mask, 0, :],
                    polygon_vertices[fan_valid_mask, fan_index, :],
                    polygon_vertices[fan_valid_mask, fan_index + 1, :],
                ],
                dim=1,
            )
        )

    if len(uv_triangle_chunks) == 0:
        return torch.zeros(
            (0, 3, 2),
            device=polygon_vertices.device,
            dtype=torch.float32,
        )
    return torch.cat(uv_triangle_chunks, dim=0).to(dtype=torch.float32).contiguous()


def _duplicate_wrapped_uv_triangles(
    uv_triangles: torch.Tensor,
) -> torch.Tensor:
    """Duplicate UV triangles across the cylindrical wrap boundary when needed.

    Args:
        uv_triangles: UV triangle soup [K, 3, 2].

    Returns:
        Wrapped UV triangle soup [Kw, 3, 2].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(uv_triangles, torch.Tensor), f"{type(uv_triangles)=}"
        assert uv_triangles.ndim == 3, f"{uv_triangles.shape=}"
        assert uv_triangles.shape[1:] == (3, 2), f"{uv_triangles.shape=}"

    _validate_inputs()

    if uv_triangles.shape[0] == 0:
        return uv_triangles.to(dtype=torch.float32).contiguous()

    wrapped_triangle_chunks = [uv_triangles.to(dtype=torch.float32).contiguous()]
    triangle_u = uv_triangles[:, :, 0]
    triangle_extends_right = triangle_u.max(dim=1).values > 1.0
    triangle_extends_left = triangle_u.min(dim=1).values < 0.0
    if torch.any(triangle_extends_right):
        shifted_right = uv_triangles[triangle_extends_right].clone()
        shifted_right[:, :, 0] = shifted_right[:, :, 0] - 1.0
        wrapped_triangle_chunks.append(shifted_right)
    if torch.any(triangle_extends_left):
        shifted_left = uv_triangles[triangle_extends_left].clone()
        shifted_left[:, :, 0] = shifted_left[:, :, 0] + 1.0
        wrapped_triangle_chunks.append(shifted_left)
    return torch.cat(wrapped_triangle_chunks, dim=0).contiguous()


def _build_uv_triangle_texel_intersections(
    uv_triangles: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Build exact UV-triangle to texel-cell intersection indices.

    Args:
        uv_triangles: UV triangle soup [K, 3, 2].
        texture_size: UV texture resolution.

    Returns:
        Covered texel indices [N, 2] in `(row, col)` order.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(uv_triangles, torch.Tensor), f"{type(uv_triangles)=}"
        assert isinstance(texture_size, int), f"{type(texture_size)=}"
        assert uv_triangles.ndim == 3, f"{uv_triangles.shape=}"
        assert uv_triangles.shape[1:] == (3, 2), f"{uv_triangles.shape=}"
        assert texture_size > 0, f"{texture_size=}"

    _validate_inputs()

    triangle_texel_vertices = uv_triangles.to(dtype=torch.float32) * float(texture_size)
    triangle_x_min = triangle_texel_vertices[:, :, 0].amin(dim=1)
    triangle_x_max = triangle_texel_vertices[:, :, 0].amax(dim=1)
    triangle_y_min = triangle_texel_vertices[:, :, 1].amin(dim=1)
    triangle_y_max = triangle_texel_vertices[:, :, 1].amax(dim=1)
    texel_x_start = torch.floor(triangle_x_min).to(dtype=torch.long)
    texel_x_end = torch.ceil(triangle_x_max).to(dtype=torch.long) - 1
    texel_y_start = torch.floor(triangle_y_min).to(dtype=torch.long)
    texel_y_end = torch.ceil(triangle_y_max).to(dtype=torch.long) - 1
    texel_x_start = texel_x_start.clamp(min=0, max=texture_size - 1)
    texel_x_end = texel_x_end.clamp(min=0, max=texture_size - 1)
    texel_y_start = texel_y_start.clamp(min=0, max=texture_size - 1)
    texel_y_end = texel_y_end.clamp(min=0, max=texture_size - 1)
    texel_x_count = (texel_x_end - texel_x_start + 1).clamp(min=0)
    texel_y_count = (texel_y_end - texel_y_start + 1).clamp(min=0)
    pair_count_per_triangle = texel_x_count * texel_y_count
    total_pair_count = int(pair_count_per_triangle.sum().item())
    if total_pair_count == 0:
        return torch.zeros(
            (0, 2),
            device=uv_triangles.device,
            dtype=torch.long,
        )

    triangle_indices = torch.arange(
        uv_triangles.shape[0],
        device=uv_triangles.device,
        dtype=torch.long,
    )
    repeated_triangle_indices = torch.repeat_interleave(
        triangle_indices,
        pair_count_per_triangle,
    )
    pair_start_offsets = (
        torch.cumsum(
            pair_count_per_triangle,
            dim=0,
        )
        - pair_count_per_triangle
    )
    repeated_pair_start_offsets = torch.repeat_interleave(
        pair_start_offsets,
        pair_count_per_triangle,
    )
    repeated_texel_x_count = torch.repeat_interleave(
        texel_x_count,
        pair_count_per_triangle,
    )
    repeated_texel_x_start = torch.repeat_interleave(
        texel_x_start,
        pair_count_per_triangle,
    )
    repeated_texel_y_start = torch.repeat_interleave(
        texel_y_start,
        pair_count_per_triangle,
    )
    pair_offsets = (
        torch.arange(
            total_pair_count,
            device=uv_triangles.device,
            dtype=torch.long,
        )
        - repeated_pair_start_offsets
    )
    local_texel_y_offset = torch.div(
        pair_offsets,
        repeated_texel_x_count,
        rounding_mode="floor",
    )
    local_texel_x_offset = pair_offsets % repeated_texel_x_count
    texel_x = repeated_texel_x_start + local_texel_x_offset
    texel_y = repeated_texel_y_start + local_texel_y_offset

    positive_overlap_mask = _compute_triangle_pixel_square_positive_area_overlap_mask(
        triangle_vertices=triangle_texel_vertices[repeated_triangle_indices],
        pixel_x=texel_x.to(dtype=torch.float32) + 0.5,
        pixel_y=texel_y.to(dtype=torch.float32) + 0.5,
    )
    if not torch.any(positive_overlap_mask):
        return torch.zeros(
            (0, 2),
            device=uv_triangles.device,
            dtype=torch.long,
        )

    covered_texel_indices = torch.stack(
        [
            texel_y[positive_overlap_mask],
            texel_x[positive_overlap_mask],
        ],
        dim=1,
    )
    return torch.unique(
        covered_texel_indices,
        dim=0,
    ).contiguous()


def _build_uv_triangle_texel_intersections_v2(
    uv_triangles: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Build approximate step-2 `v2` UV-triangle to texel-cell intersections.

    Args:
        uv_triangles: UV triangle soup [K, 3, 2].
        texture_size: UV texture resolution.

    Returns:
        Covered texel indices [N, 2] in `(row, col)` order.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(uv_triangles, torch.Tensor), (
            "Expected `uv_triangles` to be a tensor. "
            f"type(uv_triangles)={type(uv_triangles)!r}."
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an int. "
            f"type(texture_size)={type(texture_size)!r}."
        )
        assert uv_triangles.ndim == 3, (
            "Expected `uv_triangles` to have shape `[K, 3, 2]`. "
            f"uv_triangles.shape={uv_triangles.shape!r}."
        )
        assert uv_triangles.shape[1:] == (3, 2), (
            "Expected `uv_triangles` to have shape `[K, 3, 2]`. "
            f"uv_triangles.shape={uv_triangles.shape!r}."
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"texture_size={texture_size!r}."
        )

    _validate_inputs()

    def _compute_triangle_edge_function_coefficients(
        triangle_vertices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute oriented triangle edge-function coefficients and thresholds.

        Args:
            triangle_vertices: Triangle vertices [K, 3, 2] in texel space.

        Returns:
            Tuple of:
                oriented edge coefficients [K, 3, 3],
                texel-square interior thresholds [K, 3].
        """

        def _validate_triangle_vertices() -> None:
            """Validate input arguments.

            Args:
                None.

            Returns:
                None.
            """
            assert isinstance(triangle_vertices, torch.Tensor), (
                "Expected `triangle_vertices` to be a tensor. "
                f"type(triangle_vertices)={type(triangle_vertices)!r}."
            )
            assert triangle_vertices.ndim == 3, (
                "Expected `triangle_vertices` to have shape `[K, 3, 2]`. "
                f"triangle_vertices.shape={triangle_vertices.shape!r}."
            )
            assert triangle_vertices.shape[1:] == (3, 2), (
                "Expected `triangle_vertices` to have shape `[K, 3, 2]`. "
                f"triangle_vertices.shape={triangle_vertices.shape!r}."
            )

        _validate_triangle_vertices()

        next_triangle_vertices = triangle_vertices[:, [1, 2, 0], :]
        edge_a = triangle_vertices[:, :, 1] - next_triangle_vertices[:, :, 1]
        edge_b = next_triangle_vertices[:, :, 0] - triangle_vertices[:, :, 0]
        edge_c = (
            triangle_vertices[:, :, 0] * next_triangle_vertices[:, :, 1]
            - next_triangle_vertices[:, :, 0] * triangle_vertices[:, :, 1]
        )
        triangle_double_area = (
            triangle_vertices[:, 1, 0] - triangle_vertices[:, 0, 0]
        ) * (triangle_vertices[:, 2, 1] - triangle_vertices[:, 0, 1]) - (
            triangle_vertices[:, 1, 1] - triangle_vertices[:, 0, 1]
        ) * (
            triangle_vertices[:, 2, 0] - triangle_vertices[:, 0, 0]
        )
        triangle_orientation = torch.where(
            triangle_double_area >= 0.0,
            torch.ones_like(triangle_double_area),
            -torch.ones_like(triangle_double_area),
        ).reshape(-1, 1)
        oriented_edge_a = edge_a * triangle_orientation
        oriented_edge_b = edge_b * triangle_orientation
        oriented_edge_c = edge_c * triangle_orientation
        edge_function_coefficients = torch.stack(
            [oriented_edge_a, oriented_edge_b, oriented_edge_c],
            dim=2,
        ).contiguous()
        edge_thresholds = (
            0.5 * (torch.abs(oriented_edge_a) + torch.abs(oriented_edge_b))
        ).contiguous()
        return edge_function_coefficients, edge_thresholds

    if uv_triangles.dtype != torch.float32:
        assert uv_triangles.dtype in (
            torch.float16,
            torch.float64,
            torch.bfloat16,
        ), (
            "Expected `uv_triangles` dtype to be one floating-point type. "
            f"uv_triangles.dtype={uv_triangles.dtype!r}."
        )

    triangle_texel_vertices = uv_triangles.to(dtype=torch.float32) * float(texture_size)
    triangle_x_min = triangle_texel_vertices[:, :, 0].amin(dim=1)
    triangle_x_max = triangle_texel_vertices[:, :, 0].amax(dim=1)
    triangle_y_min = triangle_texel_vertices[:, :, 1].amin(dim=1)
    triangle_y_max = triangle_texel_vertices[:, :, 1].amax(dim=1)
    texel_x_start = torch.floor(triangle_x_min).to(dtype=torch.long)
    texel_x_end = torch.ceil(triangle_x_max).to(dtype=torch.long) - 1
    texel_y_start = torch.floor(triangle_y_min).to(dtype=torch.long)
    texel_y_end = torch.ceil(triangle_y_max).to(dtype=torch.long) - 1
    texel_x_start = texel_x_start.clamp(min=0, max=texture_size - 1)
    texel_x_end = texel_x_end.clamp(min=0, max=texture_size - 1)
    texel_y_start = texel_y_start.clamp(min=0, max=texture_size - 1)
    texel_y_end = texel_y_end.clamp(min=0, max=texture_size - 1)
    texel_x_count = (texel_x_end - texel_x_start + 1).clamp(min=0)
    texel_y_count = (texel_y_end - texel_y_start + 1).clamp(min=0)
    pair_count_per_triangle = texel_x_count * texel_y_count
    total_pair_count = int(pair_count_per_triangle.sum().item())
    if total_pair_count == 0:
        return torch.zeros(
            (0, 2),
            device=uv_triangles.device,
            dtype=torch.long,
        )

    (
        triangle_edge_function_coefficients,
        triangle_edge_thresholds,
    ) = _compute_triangle_edge_function_coefficients(
        triangle_vertices=triangle_texel_vertices,
    )

    triangle_indices = torch.arange(
        uv_triangles.shape[0],
        device=uv_triangles.device,
        dtype=torch.long,
    )
    repeated_triangle_indices = torch.repeat_interleave(
        triangle_indices,
        pair_count_per_triangle,
    )
    pair_start_offsets = (
        torch.cumsum(
            pair_count_per_triangle,
            dim=0,
        )
        - pair_count_per_triangle
    )
    repeated_pair_start_offsets = torch.repeat_interleave(
        pair_start_offsets,
        pair_count_per_triangle,
    )
    repeated_texel_x_count = torch.repeat_interleave(
        texel_x_count,
        pair_count_per_triangle,
    )
    repeated_texel_x_start = torch.repeat_interleave(
        texel_x_start,
        pair_count_per_triangle,
    )
    repeated_texel_y_start = torch.repeat_interleave(
        texel_y_start,
        pair_count_per_triangle,
    )
    pair_offsets = (
        torch.arange(
            total_pair_count,
            device=uv_triangles.device,
            dtype=torch.long,
        )
        - repeated_pair_start_offsets
    )
    local_texel_y_offset = torch.div(
        pair_offsets,
        repeated_texel_x_count,
        rounding_mode="floor",
    )
    local_texel_x_offset = pair_offsets % repeated_texel_x_count
    texel_x = repeated_texel_x_start + local_texel_x_offset
    texel_y = repeated_texel_y_start + local_texel_y_offset
    pixel_x = texel_x.to(dtype=torch.float32) + 0.5
    pixel_y = texel_y.to(dtype=torch.float32) + 0.5
    repeated_edge_function_coefficients = triangle_edge_function_coefficients[
        repeated_triangle_indices
    ]
    repeated_edge_thresholds = triangle_edge_thresholds[repeated_triangle_indices]
    edge_values = (
        repeated_edge_function_coefficients[:, :, 0] * pixel_x.reshape(-1, 1)
        + repeated_edge_function_coefficients[:, :, 1] * pixel_y.reshape(-1, 1)
        + repeated_edge_function_coefficients[:, :, 2]
    )
    edge_function_tolerance = 1.0e-6
    interior_mask = torch.all(
        edge_values >= repeated_edge_thresholds + edge_function_tolerance,
        dim=1,
    )
    exterior_mask = torch.any(
        edge_values <= -repeated_edge_thresholds - edge_function_tolerance,
        dim=1,
    )
    positive_overlap_mask = interior_mask.clone()
    boundary_candidate_mask = (~interior_mask) & (~exterior_mask)
    if torch.any(boundary_candidate_mask):
        boundary_candidate_indices = torch.nonzero(
            boundary_candidate_mask,
            as_tuple=False,
        ).reshape(-1)
        boundary_exact_candidate_budget = 65536
        boundary_chunk_start = 0
        while boundary_chunk_start < boundary_candidate_indices.shape[0]:
            boundary_chunk_end = min(
                boundary_chunk_start + boundary_exact_candidate_budget,
                boundary_candidate_indices.shape[0],
            )
            boundary_chunk_indices = boundary_candidate_indices[
                boundary_chunk_start:boundary_chunk_end
            ]
            boundary_positive_overlap_mask = (
                _compute_triangle_pixel_square_positive_area_overlap_mask(
                    triangle_vertices=triangle_texel_vertices[
                        repeated_triangle_indices[boundary_chunk_indices]
                    ],
                    pixel_x=pixel_x[boundary_chunk_indices],
                    pixel_y=pixel_y[boundary_chunk_indices],
                )
            )
            positive_overlap_mask[boundary_chunk_indices] = (
                boundary_positive_overlap_mask
            )
            boundary_chunk_start = boundary_chunk_end

    if not torch.any(positive_overlap_mask):
        return torch.zeros(
            (0, 2),
            device=uv_triangles.device,
            dtype=torch.long,
        )

    covered_texel_indices = torch.stack(
        [
            texel_y[positive_overlap_mask],
            texel_x[positive_overlap_mask],
        ],
        dim=1,
    )
    return torch.unique(
        covered_texel_indices,
        dim=0,
    ).contiguous()
