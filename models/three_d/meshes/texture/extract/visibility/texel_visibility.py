"""Texel-visibility helpers for mesh texture extraction.

- Main API: compute_f_visibility_mask(...)
- 1. Compute exact visible UV polygon regions from camera pixels.
  Helper function: _compute_visible_uv_polygon_regions_from_camera_pixels(...)
  - 1.1. Compute the exact visible screen-space polygon regions inside each camera pixel.
    Helper function: _compute_visible_screen_space_polygon_regions_inside_camera_pixels(...)
    - 1.1.1. Compute all face-pixel polygon intersections without considering occlusion.
      Helper function: _compute_face_pixel_polygon_intersections_without_occlusion(...)
      - 1.1.1.1. Compute the candidate camera-pixel range for each projected face.
        Local function: _compute_projected_face_pixel_bounds(...)
      - 1.1.1.2. Enumerate all candidate `(face, pixel)` pairs inside those ranges.
        Local function: _enumerate_candidate_face_pixel_pairs(...)
      - 1.1.1.3. Clip each projected face triangle to its candidate pixel square.
        Local function: _clip_face_triangles_to_pixel_squares(...)
      - 1.1.1.4. Reject degenerate overlaps and pack the surviving face-pixel polygons.
        Local function: _pack_valid_face_pixel_polygons(...)
    - 1.1.2. Compute inter-polygon occlusion and remove the hidden regions.
      Helper function: _compute_visible_screen_space_polygon_regions_with_occlusion(...)
      - 1.1.2.1. Compute affine inverse-depth coefficients for the projected faces.
        Local function: _compute_projected_face_inverse_depth_coefficients(...)
      - 1.1.2.2. Resolve the exact visible face-pixel polygons from the clipped overlaps.
        Local function: _build_exact_visible_face_pixel_polygons(...)
      - 1.1.2.3. Pack the exact visible polygons into the downstream tensor format.
        Local function: _pack_visible_polygon_outputs(...)
  - 1.2. Map those visible screen-space polygon regions into UV by barycentric interpolation on their owning faces.
    Helper function: _map_visible_screen_space_polygon_regions_to_uv(...)
    - 1.2.1. Gather the owning face screen-space, depth, and UV data for each visible polygon.
      Local function: _gather_visible_polygon_face_geometry(...)
    - 1.2.2. Compute perspective-correct UV positions for all visible polygon verts.
      Local function: _project_screen_polygon_verts_to_uv(...)
    - 1.2.3. Return the UV polygons together with the original visible polygon vertex counts.
      Local function: _pack_visible_uv_polygons(...)
- 2. Compute visible UV texels from the UV polygon regions.
  Helper function: _compute_visible_uv_texels_from_uv_polygon_regions(...)
  - 2.1. Normalize and dispatch the step-2 rasterization version.
  - 2.2. Step-2 `v1`: exact polygon-native texel construction.
    Helper function: _compute_uv_polygon_texel_contributions_v1(...)
    Implementation note: preserve cylindrical-wrap coverage before texel construction.
      Local function: _duplicate_wrap_crossing_polygons(...)
    - 2.2.1. Remove provably exterior texels from the exact-test workload.
    - 2.2.2. Accept provably interior texels without the exact predicate.
    - 2.2.3. Resolve only the remaining boundary texels with the exact positive-area predicate.
    - 2.2.4. Emit the polygon's accepted texel set.
  - 2.3. Step-2 `v2`: approximate triangulation plus edge-function rasterization.
    Helper function: _compute_uv_polygon_texel_contributions_v2(...)
    Implementation note: keep cylindrical-wrap coverage before triangulation.
      Local function: _duplicate_wrap_crossing_polygons(...)
    - 2.3.1. Triangulate wrapped convex UV polygons into a triangle soup.
    - 2.3.2. Classify bbox texels with triangle edge functions.
    - 2.3.3. Send only the boundary band through the exact triangle-square predicate.
  - 2.4. Union all polygon texel contributions into the final UV visibility mask.
"""

from typing import Dict, Tuple

import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.texture.extract.camera_geometry import (
    _verts_world_to_camera,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility_geometry import (
    _compute_convex_polygon_areas,
    build_uv_polygon_texel_intersections,
    build_uv_triangle_texel_intersections_v2,
    build_visible_face_pixel_polygons,
    camera_verts_to_pixel,
    clip_convex_polygons_to_pixel_squares,
    compute_face_inverse_depth_coefficients,
    duplicate_wrapped_uv_polygons,
    project_screen_polygons_to_face_uv,
    triangulate_convex_uv_polygons,
)
from models.three_d.meshes.texture.extract.weights.normal_weights import (
    compute_f_normals_weights,
)

# -----------------------------------------------------------------------------
# Texel-visibility hierarchy
# -----------------------------------------------------------------------------


def compute_f_visibility_mask(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_verts_uvs: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
    texel_face_map: Dict[str, torch.Tensor],
    polygon_rast_method: str = "v2",
) -> torch.Tensor:
    """Compute one-view UV-pixel visibility mask from exact camera-pixel footprints.

    Args:
        verts: Mesh verts [V, 3].
        faces: Mesh faces [F, 3].
        face_verts_uvs: Seam-safe per-face UV triangles [F, 3, 2].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        texel_face_map: Texel -> mesh-face correspondence dict from
            `build_texel_face_map` with keys `"texel_face_index"` [T, T] int64
            (`-1` at unoccupied texels) and `"texel_face_barycentric"`
            [T, T, 3] float32.
        polygon_rast_method: Step-2 polygon rasterization method, `"v1"` or `"v2"`.

    Returns:
        Float tensor [1, T, T, 1] with values in {0, 1}.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        assert isinstance(verts, torch.Tensor), f"{type(verts)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(face_verts_uvs, torch.Tensor), f"{type(face_verts_uvs)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert isinstance(texel_face_map, dict), f"{type(texel_face_map)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        assert verts.ndim == 2, f"{verts.shape=}"
        assert verts.shape[1] == 3, f"{verts.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert faces.dtype == torch.long, f"{faces.dtype=}"
        assert face_verts_uvs.ndim == 3, f"{face_verts_uvs.shape=}"
        assert face_verts_uvs.shape == (faces.shape[0], 3, 2), (
            "Expected `face_verts_uvs` to align with `faces` as `[F, 3, 2]`. "
            f"{face_verts_uvs.shape=} {faces.shape=}"
        )
        assert len(camera) == 1, f"{len(camera)=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert "texel_face_index" in texel_face_map, f"{texel_face_map.keys()=}"
        assert isinstance(
            texel_face_map["texel_face_index"], torch.Tensor
        ), f"{type(texel_face_map['texel_face_index'])=}"
        assert (
            texel_face_map["texel_face_index"].ndim == 2
        ), f"{texel_face_map['texel_face_index'].shape=}"
        assert (
            texel_face_map["texel_face_index"].shape[0]
            == texel_face_map["texel_face_index"].shape[1]
        ), f"{texel_face_map['texel_face_index'].shape=}"
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )

    _validate_inputs()

    with torch.no_grad():
        verts_camera = _verts_world_to_camera(
            verts=verts,
            camera=camera,
        )
        texel_face_index = texel_face_map["texel_face_index"]
        assert verts.device == faces.device, (
            "Expected `verts` and `faces` to share a device. "
            f"{verts.device=} {faces.device=}"
        )
        assert verts.device == texel_face_index.device, (
            "Expected `verts` and `texel_face_index` to share a device. "
            f"{verts.device=} {texel_face_index.device=}"
        )
        assert verts.device == face_verts_uvs.device, (
            "Expected `verts` and `face_verts_uvs` to share a device. "
            f"{verts.device=} {face_verts_uvs.device=}"
        )

        texture_size = int(texel_face_index.shape[0])
        uv_occupancy_mask = (
            (texel_face_index >= 0)
            .to(dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .contiguous()
        )
        face_front_facing_mask = (
            compute_f_normals_weights(
                mesh=Mesh(verts=verts, faces=faces),
                camera=camera,
                weights_cfg={"weights": "normals"},
            )
            > 0.0
        )
        (
            uv_polygon_verts,
            uv_polygon_vertex_counts,
        ) = _compute_visible_uv_polygon_regions_from_camera_pixels(
            verts_camera=verts_camera,
            faces=faces,
            intrinsics=camera[0].intrinsics,
            image_height=image_height,
            image_width=image_width,
            face_front_facing_mask=face_front_facing_mask,
            camera_face_verts_uvs=face_verts_uvs,
        )
        uv_visible = _compute_visible_uv_texels_from_uv_polygon_regions(
            uv_polygon_verts=uv_polygon_verts,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=texture_size,
            polygon_rast_method=polygon_rast_method,
        )
        return (uv_visible * uv_occupancy_mask).contiguous()


def _compute_visible_uv_polygon_regions_from_camera_pixels(
    verts_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    face_front_facing_mask: torch.Tensor,
    camera_face_verts_uvs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact visible UV polygon regions from camera pixels.

    Args:
        verts_camera: Camera-space verts [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        face_front_facing_mask: Binary front-facing face mask [F].
        camera_face_verts_uvs: Seam-safe per-face UV coordinates [F, 3, 2].

    Returns:
        Tuple of:
            visible UV polygons [N, Vmax, 2],
            visible UV polygon vertex counts [N].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts_camera, torch.Tensor), f"{type(verts_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert isinstance(
            face_front_facing_mask, torch.Tensor
        ), f"{type(face_front_facing_mask)=}"
        assert isinstance(
            camera_face_verts_uvs, torch.Tensor
        ), f"{type(camera_face_verts_uvs)=}"
        assert verts_camera.ndim == 2, f"{verts_camera.shape=}"
        assert verts_camera.shape[1] == 3, f"{verts_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert face_front_facing_mask.shape == (
            faces.shape[0],
        ), f"{face_front_facing_mask.shape=} {faces.shape=}"
        assert camera_face_verts_uvs.shape == (
            faces.shape[0],
            3,
            2,
        ), f"{camera_face_verts_uvs.shape=} {faces.shape=}"

    _validate_inputs()

    vertex_pixels = camera_verts_to_pixel(
        verts_camera=verts_camera,
        intrinsics=intrinsics,
    )
    all_face_screen_verts = vertex_pixels[faces]
    all_face_vertex_depth = verts_camera[faces][:, :, 2]
    front_face_indices = torch.nonzero(
        face_front_facing_mask,
        as_tuple=False,
    ).reshape(-1)
    if front_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, 3, 2),
                device=verts_camera.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=verts_camera.device,
                dtype=torch.long,
            ),
        )

    face_screen_verts = all_face_screen_verts[front_face_indices]
    face_vertex_depth = all_face_vertex_depth[front_face_indices]
    face_verts_uvs = camera_face_verts_uvs[front_face_indices]
    (
        visible_screen_polygon_verts,
        visible_screen_polygon_vertex_counts,
        visible_screen_polygon_face_indices,
    ) = _compute_visible_screen_space_polygon_regions_inside_camera_pixels(
        face_screen_verts=face_screen_verts,
        face_vertex_depth=face_vertex_depth,
        image_height=image_height,
        image_width=image_width,
    )
    if visible_screen_polygon_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, visible_screen_polygon_verts.shape[1], 2),
                device=verts_camera.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=verts_camera.device,
                dtype=torch.long,
            ),
        )

    return _map_visible_screen_space_polygon_regions_to_uv(
        visible_screen_polygon_verts=visible_screen_polygon_verts,
        visible_screen_polygon_vertex_counts=visible_screen_polygon_vertex_counts,
        visible_screen_polygon_face_indices=visible_screen_polygon_face_indices,
        face_screen_verts=face_screen_verts,
        face_vertex_depth=face_vertex_depth,
        face_verts_uvs=face_verts_uvs,
    )


def _compute_visible_screen_space_polygon_regions_inside_camera_pixels(
    face_screen_verts: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact visible screen-space polygon regions inside each camera pixel.

    Args:
        face_screen_verts: Projected triangle verts [F, 3, 2].
        face_vertex_depth: Camera-space vertex depths [F, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Tuple of:
            visible screen polygons [N, Vmax, 2],
            visible polygon vertex counts [N],
            visible local face indices [N].
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
            face_screen_verts, torch.Tensor
        ), f"{type(face_screen_verts)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert face_screen_verts.ndim == 3, f"{face_screen_verts.shape=}"
        assert face_screen_verts.shape[1:] == (
            3,
            2,
        ), f"{face_screen_verts.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_verts.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_verts.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    (
        clipped_polygon_verts,
        clipped_polygon_vertex_counts,
        clipped_pixel_indices,
        clipped_face_indices,
    ) = _compute_face_pixel_polygon_intersections_without_occlusion(
        face_screen_verts=face_screen_verts,
        image_height=image_height,
        image_width=image_width,
    )
    if clipped_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, clipped_polygon_verts.shape[1], 2),
                device=face_screen_verts.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=face_screen_verts.device,
                dtype=torch.long,
            ),
            torch.zeros(
                (0,),
                device=face_screen_verts.device,
                dtype=torch.long,
            ),
        )

    return _compute_visible_screen_space_polygon_regions_with_occlusion(
        clipped_polygon_verts=clipped_polygon_verts,
        clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
        clipped_pixel_indices=clipped_pixel_indices,
        clipped_face_indices=clipped_face_indices,
        face_screen_verts=face_screen_verts,
        face_vertex_depth=face_vertex_depth,
        image_height=image_height,
        image_width=image_width,
    )


def _compute_face_pixel_polygon_intersections_without_occlusion(
    face_screen_verts: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all face-pixel polygon intersections without considering occlusion.

    Args:
        face_screen_verts: Projected triangle verts [F, 3, 2].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Tuple of:
            clipped polygons [P, Vmax, 2],
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
            face_screen_verts, torch.Tensor
        ), f"{type(face_screen_verts)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert face_screen_verts.ndim == 3, f"{face_screen_verts.shape=}"
        assert face_screen_verts.shape[1:] == (
            3,
            2,
        ), f"{face_screen_verts.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    def _compute_projected_face_pixel_bounds() -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Compute candidate pixel bounds for each projected face.

        Args:
            None.

        Returns:
            Tuple of pixel-range tensors and the total candidate pair count.
        """
        face_x_min = face_screen_verts[:, :, 0].amin(dim=1)
        face_x_max = face_screen_verts[:, :, 0].amax(dim=1)
        face_y_min = face_screen_verts[:, :, 1].amin(dim=1)
        face_y_max = face_screen_verts[:, :, 1].amax(dim=1)
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
        return (
            pixel_x_start,
            pixel_y_start,
            pixel_x_count,
            pixel_y_count,
            pixel_x_end,
            pixel_y_end,
            pair_count_per_face,
            total_pair_count,
        )

    def _enumerate_candidate_face_pixel_pairs(
        pair_count_per_face: torch.Tensor,
        pixel_x_start: torch.Tensor,
        pixel_y_start: torch.Tensor,
        pixel_x_count: torch.Tensor,
        total_pair_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enumerate all candidate face-pixel pairs.

        Args:
            pair_count_per_face: Candidate pair counts per face [F].
            pixel_x_start: Start x per face [F].
            pixel_y_start: Start y per face [F].
            pixel_x_count: Candidate x-count per face [F].
            total_pair_count: Total pair count.

        Returns:
            Repeated face indices and pixel coordinates for each pair.
        """
        local_face_indices = torch.arange(
            face_screen_verts.shape[0],
            device=face_screen_verts.device,
            dtype=torch.long,
        )
        repeated_face_indices = torch.repeat_interleave(
            local_face_indices,
            pair_count_per_face,
        )
        pair_start_offsets = (
            torch.cumsum(pair_count_per_face, dim=0) - pair_count_per_face
        )
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
                device=face_screen_verts.device,
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
        return repeated_face_indices, pixel_x, pixel_y

    def _clip_face_triangles_to_pixel_squares(
        repeated_face_indices: torch.Tensor,
        pixel_x: torch.Tensor,
        pixel_y: torch.Tensor,
        total_pair_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clip projected face triangles to candidate pixel squares.

        Args:
            repeated_face_indices: Face index per candidate pair [P].
            pixel_x: Pixel x per candidate pair [P].
            pixel_y: Pixel y per candidate pair [P].
            total_pair_count: Total pair count.

        Returns:
            Clipped polygon verts and counts for each candidate pair.
        """
        polygon_verts = torch.zeros(
            (total_pair_count, 8, 2),
            device=face_screen_verts.device,
            dtype=torch.float32,
        )
        polygon_verts[:, :3, :] = face_screen_verts[repeated_face_indices].to(
            dtype=torch.float32
        )
        polygon_vertex_counts = torch.full(
            (total_pair_count,),
            fill_value=3,
            device=face_screen_verts.device,
            dtype=torch.long,
        )
        return clip_convex_polygons_to_pixel_squares(
            polygon_verts=polygon_verts,
            polygon_vertex_counts=polygon_vertex_counts,
            pixel_x=pixel_x.to(dtype=torch.float32),
            pixel_y=pixel_y.to(dtype=torch.float32),
        )

    def _pack_valid_face_pixel_polygons(
        clipped_polygon_verts: torch.Tensor,
        clipped_polygon_vertex_counts: torch.Tensor,
        pixel_x: torch.Tensor,
        pixel_y: torch.Tensor,
        repeated_face_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reject degenerate overlaps and pack the surviving polygons.

        Args:
            clipped_polygon_verts: Clipped polygon verts [P, Vmax, 2].
            clipped_polygon_vertex_counts: Clipped polygon vertex counts [P].
            pixel_x: Pixel x per candidate pair [P].
            pixel_y: Pixel y per candidate pair [P].
            repeated_face_indices: Face index per candidate pair [P].

        Returns:
            Packed valid polygons, counts, pixel indices, and face indices.
        """
        clipped_polygon_area = _compute_convex_polygon_areas(
            polygon_verts=clipped_polygon_verts,
            polygon_vertex_counts=clipped_polygon_vertex_counts,
        )
        polygon_valid_mask = (clipped_polygon_vertex_counts >= 3) & (
            clipped_polygon_area > 1.0e-12
        )
        return (
            clipped_polygon_verts[polygon_valid_mask].contiguous(),
            clipped_polygon_vertex_counts[polygon_valid_mask].contiguous(),
            torch.stack(
                [pixel_y[polygon_valid_mask], pixel_x[polygon_valid_mask]],
                dim=1,
            ).contiguous(),
            repeated_face_indices[polygon_valid_mask].contiguous(),
        )

    (
        pixel_x_start,
        pixel_y_start,
        pixel_x_count,
        _pixel_y_count,
        _pixel_x_end,
        _pixel_y_end,
        pair_count_per_face,
        total_pair_count,
    ) = _compute_projected_face_pixel_bounds()
    if total_pair_count == 0:
        empty_long = torch.zeros(
            (0,),
            device=face_screen_verts.device,
            dtype=torch.long,
        )
        return (
            torch.zeros(
                (0, 8, 2),
                device=face_screen_verts.device,
                dtype=torch.float32,
            ),
            empty_long,
            torch.zeros(
                (0, 2),
                device=face_screen_verts.device,
                dtype=torch.long,
            ),
            empty_long,
        )

    repeated_face_indices, pixel_x, pixel_y = _enumerate_candidate_face_pixel_pairs(
        pair_count_per_face=pair_count_per_face,
        pixel_x_start=pixel_x_start,
        pixel_y_start=pixel_y_start,
        pixel_x_count=pixel_x_count,
        total_pair_count=total_pair_count,
    )
    clipped_polygon_verts, clipped_polygon_vertex_counts = (
        _clip_face_triangles_to_pixel_squares(
            repeated_face_indices=repeated_face_indices,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            total_pair_count=total_pair_count,
        )
    )
    return _pack_valid_face_pixel_polygons(
        clipped_polygon_verts=clipped_polygon_verts,
        clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        repeated_face_indices=repeated_face_indices,
    )


def _compute_visible_screen_space_polygon_regions_with_occlusion(
    clipped_polygon_verts: torch.Tensor,
    clipped_polygon_vertex_counts: torch.Tensor,
    clipped_pixel_indices: torch.Tensor,
    clipped_face_indices: torch.Tensor,
    face_screen_verts: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute inter-polygon occlusion and remove hidden screen-space regions.

    Args:
        clipped_polygon_verts: Face-pixel polygons [P, Vmax, 2].
        clipped_polygon_vertex_counts: Valid polygon vertex counts [P].
        clipped_pixel_indices: Pixel indices [P, 2] in `(y, x)` order.
        clipped_face_indices: Local face indices [P].
        face_screen_verts: Projected triangle verts [F, 3, 2].
        face_vertex_depth: Camera-space vertex depths [F, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Tuple of:
            visible screen polygons [N, Vmax, 2],
            visible polygon vertex counts [N],
            visible local face indices [N].
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
            clipped_polygon_verts, torch.Tensor
        ), f"{type(clipped_polygon_verts)=}"
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
            face_screen_verts, torch.Tensor
        ), f"{type(face_screen_verts)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert clipped_polygon_verts.ndim == 3, f"{clipped_polygon_verts.shape=}"
        assert (
            clipped_polygon_verts.shape[2] == 2
        ), f"{clipped_polygon_verts.shape=}"
        assert clipped_polygon_vertex_counts.shape == (
            clipped_polygon_verts.shape[0],
        ), f"{clipped_polygon_vertex_counts.shape=} {clipped_polygon_verts.shape=}"
        assert clipped_pixel_indices.shape == (
            clipped_polygon_verts.shape[0],
            2,
        ), f"{clipped_pixel_indices.shape=} {clipped_polygon_verts.shape=}"
        assert clipped_face_indices.shape == (
            clipped_polygon_verts.shape[0],
        ), f"{clipped_face_indices.shape=} {clipped_polygon_verts.shape=}"
        assert face_screen_verts.ndim == 3, f"{face_screen_verts.shape=}"
        assert face_screen_verts.shape[1:] == (
            3,
            2,
        ), f"{face_screen_verts.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_verts.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_verts.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    def _compute_projected_face_inverse_depth_coefficients() -> torch.Tensor:
        """Compute affine inverse-depth coefficients for the projected faces.

        Args:
            None.

        Returns:
            Inverse-depth coefficients [F, 3].
        """
        return compute_face_inverse_depth_coefficients(
            face_screen_verts=face_screen_verts,
            face_vertex_depth=face_vertex_depth,
        )

    def _build_exact_visible_face_pixel_polygons(
        face_inverse_depth_coefficients: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve exact visible face-pixel polygons from clipped overlaps.

        Args:
            face_inverse_depth_coefficients: Inverse-depth coefficients [F, 3].

        Returns:
            Visible polygon verts, counts, and local face indices.
        """
        return build_visible_face_pixel_polygons(
            clipped_polygon_verts=clipped_polygon_verts,
            clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
            clipped_pixel_indices=clipped_pixel_indices,
            clipped_face_indices=clipped_face_indices,
            face_inverse_depth_coefficients=face_inverse_depth_coefficients,
        )

    def _pack_visible_polygon_outputs(
        visible_polygon_verts: torch.Tensor,
        visible_polygon_vertex_counts: torch.Tensor,
        visible_polygon_face_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack exact visible polygons into the downstream tensor format.

        Args:
            visible_polygon_verts: Visible polygons [N, Vmax, 2].
            visible_polygon_vertex_counts: Visible polygon vertex counts [N].
            visible_polygon_face_indices: Visible local face indices [N].

        Returns:
            Visible polygon verts, counts, and local face indices.
        """
        return (
            visible_polygon_verts.contiguous(),
            visible_polygon_vertex_counts.contiguous(),
            visible_polygon_face_indices.contiguous(),
        )

    if clipped_polygon_verts.shape[0] == 0:
        return (
            torch.zeros(
                (0, clipped_polygon_verts.shape[1], 2),
                device=clipped_polygon_verts.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=clipped_polygon_verts.device,
                dtype=torch.long,
            ),
            torch.zeros(
                (0,),
                device=clipped_polygon_verts.device,
                dtype=torch.long,
            ),
        )

    face_inverse_depth_coefficients = _compute_projected_face_inverse_depth_coefficients()
    (
        visible_polygon_verts,
        visible_polygon_vertex_counts,
        visible_polygon_face_indices,
    ) = _build_exact_visible_face_pixel_polygons(
        face_inverse_depth_coefficients=face_inverse_depth_coefficients,
    )
    return _pack_visible_polygon_outputs(
        visible_polygon_verts=visible_polygon_verts,
        visible_polygon_vertex_counts=visible_polygon_vertex_counts,
        visible_polygon_face_indices=visible_polygon_face_indices,
    )


def _map_visible_screen_space_polygon_regions_to_uv(
    visible_screen_polygon_verts: torch.Tensor,
    visible_screen_polygon_vertex_counts: torch.Tensor,
    visible_screen_polygon_face_indices: torch.Tensor,
    face_screen_verts: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    face_verts_uvs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map visible screen-space polygon regions into UV.

    Args:
        visible_screen_polygon_verts: Visible screen polygons [N, Vmax, 2].
        visible_screen_polygon_vertex_counts: Valid polygon vertex counts [N].
        visible_screen_polygon_face_indices: Local face index for each visible polygon [N].
        face_screen_verts: Projected triangle verts [F, 3, 2].
        face_vertex_depth: Camera-space vertex depths [F, 3].
        face_verts_uvs: Seam-safe per-face UV coordinates [F, 3, 2].

    Returns:
        Tuple of:
            visible UV polygons [N, Vmax, 2],
            visible UV polygon vertex counts [N].
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
            visible_screen_polygon_verts, torch.Tensor
        ), f"{type(visible_screen_polygon_verts)=}"
        assert isinstance(
            visible_screen_polygon_vertex_counts, torch.Tensor
        ), f"{type(visible_screen_polygon_vertex_counts)=}"
        assert isinstance(
            visible_screen_polygon_face_indices, torch.Tensor
        ), f"{type(visible_screen_polygon_face_indices)=}"
        assert isinstance(
            face_screen_verts, torch.Tensor
        ), f"{type(face_screen_verts)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(face_verts_uvs, torch.Tensor), f"{type(face_verts_uvs)=}"
        assert (
            visible_screen_polygon_verts.ndim == 3
        ), f"{visible_screen_polygon_verts.shape=}"
        assert (
            visible_screen_polygon_verts.shape[2] == 2
        ), f"{visible_screen_polygon_verts.shape=}"
        assert visible_screen_polygon_vertex_counts.shape == (
            visible_screen_polygon_verts.shape[0],
        ), f"{visible_screen_polygon_vertex_counts.shape=} {visible_screen_polygon_verts.shape=}"
        assert visible_screen_polygon_face_indices.shape == (
            visible_screen_polygon_verts.shape[0],
        ), f"{visible_screen_polygon_face_indices.shape=} {visible_screen_polygon_verts.shape=}"
        assert face_screen_verts.ndim == 3, f"{face_screen_verts.shape=}"
        assert face_screen_verts.shape[1:] == (
            3,
            2,
        ), f"{face_screen_verts.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_verts.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_verts.shape=}"
        assert face_verts_uvs.shape == (
            face_screen_verts.shape[0],
            3,
            2,
        ), f"{face_verts_uvs.shape=} {face_screen_verts.shape=}"

    _validate_inputs()

    def _gather_visible_polygon_face_geometry() -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gather owning-face geometry for each visible polygon.

        Args:
            None.

        Returns:
            Screen-space verts, depths, and UVs for each visible polygon.
        """
        return (
            face_screen_verts[visible_screen_polygon_face_indices].contiguous(),
            face_vertex_depth[visible_screen_polygon_face_indices].contiguous(),
            face_verts_uvs[visible_screen_polygon_face_indices].contiguous(),
        )

    def _project_screen_polygon_verts_to_uv(
        polygon_face_screen_verts: torch.Tensor,
        polygon_face_vertex_depth: torch.Tensor,
        polygon_face_verts_uvs: torch.Tensor,
    ) -> torch.Tensor:
        """Project visible screen polygons into UV.

        Args:
            polygon_face_screen_verts: Owning face screen verts [N, 3, 2].
            polygon_face_vertex_depth: Owning face depths [N, 3].
            polygon_face_verts_uvs: Owning face UVs [N, 3, 2].

        Returns:
            Visible UV polygons [N, Vmax, 2].
        """
        return project_screen_polygons_to_face_uv(
            polygon_verts=visible_screen_polygon_verts,
            face_screen_verts=polygon_face_screen_verts,
            face_vertex_depth=polygon_face_vertex_depth,
            face_verts_uvs=polygon_face_verts_uvs,
        )

    def _pack_visible_uv_polygons(
        uv_polygon_verts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack UV polygons with their original vertex counts.

        Args:
            uv_polygon_verts: Visible UV polygons [N, Vmax, 2].

        Returns:
            UV polygons and their vertex counts.
        """
        return (
            uv_polygon_verts.contiguous(),
            visible_screen_polygon_vertex_counts.contiguous(),
        )

    (
        polygon_face_screen_verts,
        polygon_face_vertex_depth,
        polygon_face_verts_uvs,
    ) = _gather_visible_polygon_face_geometry()
    uv_polygon_verts = _project_screen_polygon_verts_to_uv(
        polygon_face_screen_verts=polygon_face_screen_verts,
        polygon_face_vertex_depth=polygon_face_vertex_depth,
        polygon_face_verts_uvs=polygon_face_verts_uvs,
    )
    return _pack_visible_uv_polygons(
        uv_polygon_verts=uv_polygon_verts,
    )


def _compute_visible_uv_texels_from_uv_polygon_regions(
    uv_polygon_verts: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
    polygon_rast_method: str = "v2",
) -> torch.Tensor:
    """Compute visible UV texels from the UV polygon regions.

    Args:
        uv_polygon_verts: Visible UV polygons [N, Vmax, 2].
        uv_polygon_vertex_counts: Valid polygon vertex counts [N].
        texture_size: UV texture resolution.
        polygon_rast_method: Step-2 polygon rasterization method, `"v1"` or `"v2"`.

    Returns:
        UV visibility mask [1, T, T, 1].
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
            uv_polygon_verts, torch.Tensor
        ), f"{type(uv_polygon_verts)=}"
        assert isinstance(
            uv_polygon_vertex_counts, torch.Tensor
        ), f"{type(uv_polygon_vertex_counts)=}"
        assert isinstance(texture_size, int), f"{type(texture_size)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        assert uv_polygon_verts.ndim == 3, f"{uv_polygon_verts.shape=}"
        assert uv_polygon_verts.shape[2] == 2, f"{uv_polygon_verts.shape=}"
        assert uv_polygon_vertex_counts.shape == (
            uv_polygon_verts.shape[0],
        ), f"{uv_polygon_vertex_counts.shape=} {uv_polygon_verts.shape=}"
        assert texture_size > 0, f"{texture_size=}"
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )

    _validate_inputs()

    if polygon_rast_method == "v1":
        covered_texel_indices = _compute_uv_polygon_texel_contributions_v1(
            uv_polygon_verts=uv_polygon_verts,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=texture_size,
        )
    else:
        covered_texel_indices = _compute_uv_polygon_texel_contributions_v2(
            uv_polygon_verts=uv_polygon_verts,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=texture_size,
        )
    uv_mask = torch.zeros(
        (1, texture_size, texture_size, 1),
        device=uv_polygon_verts.device,
        dtype=torch.float32,
    )
    if covered_texel_indices.shape[0] == 0:
        return uv_mask

    uv_mask[
        0,
        covered_texel_indices[:, 0],
        covered_texel_indices[:, 1],
        0,
    ] = 1.0
    return uv_mask.contiguous()


def _compute_uv_polygon_texel_contributions_v1(
    uv_polygon_verts: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Construct exact step-2 `v1` texel contributions for visible UV polygons.

    Args:
        uv_polygon_verts: Visible UV polygons [N, Vmax, 2].
        uv_polygon_vertex_counts: Valid polygon vertex counts [N].
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
        assert isinstance(
            uv_polygon_verts, torch.Tensor
        ), f"{type(uv_polygon_verts)=}"
        assert isinstance(
            uv_polygon_vertex_counts, torch.Tensor
        ), f"{type(uv_polygon_vertex_counts)=}"
        assert uv_polygon_verts.ndim == 3, f"{uv_polygon_verts.shape=}"
        assert uv_polygon_verts.shape[2] == 2, f"{uv_polygon_verts.shape=}"
        assert uv_polygon_vertex_counts.shape == (
            uv_polygon_verts.shape[0],
        ), f"{uv_polygon_vertex_counts.shape=} {uv_polygon_verts.shape=}"
        assert isinstance(texture_size, int), f"{type(texture_size)=}"
        assert texture_size > 0, f"{texture_size=}"

    _validate_inputs()

    def _duplicate_wrap_crossing_polygons() -> Tuple[torch.Tensor, torch.Tensor]:
        """Duplicate wrap-crossing polygons so the cylindrical UV union is preserved.

        Args:
            None.

        Returns:
            Tuple of:
                wrapped UV polygons [Nw, Vmax, 2],
                wrapped UV polygon vertex counts [Nw].
        """
        return duplicate_wrapped_uv_polygons(
            uv_polygon_verts=uv_polygon_verts,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
        )

    (
        wrapped_uv_polygon_verts,
        wrapped_uv_polygon_vertex_counts,
    ) = _duplicate_wrap_crossing_polygons()
    return build_uv_polygon_texel_intersections(
        uv_polygon_verts=wrapped_uv_polygon_verts,
        uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
        texture_size=texture_size,
    )


def _compute_uv_polygon_texel_contributions_v2(
    uv_polygon_verts: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Construct approximate step-2 `v2` texel contributions for visible UV polygons.

    Args:
        uv_polygon_verts: Visible UV polygons [N, Vmax, 2].
        uv_polygon_vertex_counts: Valid polygon vertex counts [N].
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
        assert isinstance(uv_polygon_verts, torch.Tensor), (
            "Expected `uv_polygon_verts` to be a tensor. "
            f"Got {type(uv_polygon_verts)=}."
        )
        assert isinstance(uv_polygon_vertex_counts, torch.Tensor), (
            "Expected `uv_polygon_vertex_counts` to be a tensor. "
            f"Got {type(uv_polygon_vertex_counts)=}."
        )
        assert uv_polygon_verts.ndim == 3, (
            "Expected `uv_polygon_verts` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_verts.shape=}."
        )
        assert uv_polygon_verts.shape[2] == 2, (
            "Expected `uv_polygon_verts` to have shape `[N, Vmax, 2]`. "
            f"Got {uv_polygon_verts.shape=}."
        )
        assert uv_polygon_vertex_counts.shape == (uv_polygon_verts.shape[0],), (
            "Expected `uv_polygon_vertex_counts` to align with polygon count. "
            f"{uv_polygon_vertex_counts.shape=} {uv_polygon_verts.shape=}."
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an int. " f"Got {type(texture_size)=}."
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"Got {texture_size=}."
        )

    _validate_inputs()

    def _duplicate_wrap_crossing_polygons() -> Tuple[torch.Tensor, torch.Tensor]:
        """Duplicate wrap-crossing polygons so the cylindrical UV union is preserved.

        Args:
            None.

        Returns:
            Tuple of:
                wrapped UV polygons [Nw, Vmax, 2],
                wrapped UV polygon vertex counts [Nw].
        """
        return duplicate_wrapped_uv_polygons(
            uv_polygon_verts=uv_polygon_verts,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
        )

    def _triangulate_wrapped_uv_polygons(
        wrapped_uv_polygon_verts: torch.Tensor,
        wrapped_uv_polygon_vertex_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Triangulate wrapped convex UV polygons into a triangle soup.

        Args:
            wrapped_uv_polygon_verts: Wrapped UV polygons [Nw, Vmax, 2].
            wrapped_uv_polygon_vertex_counts: Wrapped UV polygon vertex counts [Nw].

        Returns:
            Wrapped UV triangle soup [K, 3, 2].
        """

        def _validate_inputs() -> None:
            """Validate input arguments.

            Args:
                None.

            Returns:
                None.
            """
            assert isinstance(wrapped_uv_polygon_verts, torch.Tensor), (
                "Expected `wrapped_uv_polygon_verts` to be a tensor. "
                f"Got {type(wrapped_uv_polygon_verts)=}."
            )
            assert isinstance(wrapped_uv_polygon_vertex_counts, torch.Tensor), (
                "Expected `wrapped_uv_polygon_vertex_counts` to be a tensor. "
                f"Got {type(wrapped_uv_polygon_vertex_counts)=}."
            )
            assert wrapped_uv_polygon_verts.ndim == 3, (
                "Expected `wrapped_uv_polygon_verts` to be rank-3. "
                f"{wrapped_uv_polygon_verts.shape=}."
            )
            assert wrapped_uv_polygon_verts.shape[2] == 2, (
                "Expected `wrapped_uv_polygon_verts` to end with UV pairs. "
                f"{wrapped_uv_polygon_verts.shape=}."
            )
            assert wrapped_uv_polygon_vertex_counts.shape == (
                wrapped_uv_polygon_verts.shape[0],
            ), (
                "Expected wrapped vertex counts to align with polygon count. "
                f"{wrapped_uv_polygon_vertex_counts.shape=} "
                f"{wrapped_uv_polygon_verts.shape=}."
            )

        _validate_inputs()

        return triangulate_convex_uv_polygons(
            polygon_verts=wrapped_uv_polygon_verts,
            polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
        )

    (
        wrapped_uv_polygon_verts,
        wrapped_uv_polygon_vertex_counts,
    ) = _duplicate_wrap_crossing_polygons()
    wrapped_uv_triangles = _triangulate_wrapped_uv_polygons(
        wrapped_uv_polygon_verts=wrapped_uv_polygon_verts,
        wrapped_uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
    )
    return build_uv_triangle_texel_intersections_v2(
        uv_triangles=wrapped_uv_triangles,
        texture_size=texture_size,
    )
