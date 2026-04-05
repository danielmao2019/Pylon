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
        Local function: _compute_face_inverse_depth_coefficients(...)
      - 1.1.2.2. Resolve the exact visible face-pixel polygons from the clipped overlaps.
        Local function: _build_exact_visible_face_pixel_polygons(...)
      - 1.1.2.3. Pack the exact visible polygons into the downstream tensor format.
        Local function: _pack_visible_polygon_outputs(...)
  - 1.2. Map those visible screen-space polygon regions into UV by barycentric interpolation on their owning faces.
    Helper function: _map_visible_screen_space_polygon_regions_to_uv(...)
    - 1.2.1. Gather the owning face screen-space, depth, and UV data for each visible polygon.
      Local function: _gather_visible_polygon_face_geometry(...)
    - 1.2.2. Compute perspective-correct UV positions for all visible polygon vertices.
      Local function: _project_screen_polygon_vertices_to_uv(...)
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

import models.three_d.meshes.texture.extract.visibility.texel_visibility_geometry as _geom
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.texture.extract.camera_geometry import (
    _vertices_world_to_camera,
)
from models.three_d.meshes.texture.extract.normal_weights import (
    _compute_f_normals_weights,
)

# -----------------------------------------------------------------------------
# Texel-visibility hierarchy
# -----------------------------------------------------------------------------


def compute_f_visibility_mask(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
    uv_rasterization_data: Dict[str, torch.Tensor],
    polygon_rast_method: str = "v2",
) -> torch.Tensor:
    """Compute one-view UV-pixel visibility mask from exact camera-pixel footprints.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        uv_rasterization_data: Precomputed UV rasterization tensors.
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
        # Input validations
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert isinstance(
            uv_rasterization_data, dict
        ), f"{type(uv_rasterization_data)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert faces.dtype == torch.long, f"{faces.dtype=}"
        assert len(camera) == 1, f"{len(camera)=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert "uv_mask" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
        assert (
            "camera_attr_vertex_uv" in uv_rasterization_data
        ), f"{uv_rasterization_data.keys()=}"
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )

    _validate_inputs()

    with torch.no_grad():
        vertices_camera = _vertices_world_to_camera(
            vertices=vertices,
            camera=camera,
        )
        uv_mask = uv_rasterization_data["uv_mask"]
        camera_attr_vertex_uv = uv_rasterization_data["camera_attr_vertex_uv"]
        assert isinstance(uv_mask, torch.Tensor), f"{type(uv_mask)=}"
        assert isinstance(camera_attr_vertex_uv, torch.Tensor), (
            "Expected `camera_attr_vertex_uv` to be a tensor. "
            f"Got {type(camera_attr_vertex_uv)=}."
        )
        assert vertices.device == faces.device, (
            "Expected `vertices` and `faces` to share a device. "
            f"Got {vertices.device=} {faces.device=}."
        )
        assert vertices.device == uv_mask.device, (
            "Expected `vertices` and `uv_mask` to share a device. "
            f"Got {vertices.device=} {uv_mask.device=}."
        )
        assert vertices.device == camera_attr_vertex_uv.device, (
            "Expected `vertices` and `camera_attr_vertex_uv` to share a device. "
            f"Got {vertices.device=} {camera_attr_vertex_uv.device=}."
        )

        face_front_facing_mask = (
            _compute_f_normals_weights(
                mesh=Mesh(vertices=vertices, faces=faces),
                camera=camera,
                weights_cfg={"weights": "normals"},
            )
            > 0.0
        )
        (
            uv_polygon_vertices,
            uv_polygon_vertex_counts,
        ) = _compute_visible_uv_polygon_regions_from_camera_pixels(
            vertices_camera=vertices_camera,
            faces=faces,
            intrinsics=camera[0].intrinsics,
            image_height=image_height,
            image_width=image_width,
            face_front_facing_mask=face_front_facing_mask,
            camera_face_vertex_uv=camera_attr_vertex_uv.reshape(-1, 3, 2),
        )
        uv_visible = _compute_visible_uv_texels_from_uv_polygon_regions(
            uv_polygon_vertices=uv_polygon_vertices,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=int(uv_mask.shape[1]),
            polygon_rast_method=polygon_rast_method,
        )
        return (uv_visible * uv_mask).contiguous()


def _compute_visible_uv_polygon_regions_from_camera_pixels(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    face_front_facing_mask: torch.Tensor,
    camera_face_vertex_uv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact visible UV polygon regions from camera pixels.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        face_front_facing_mask: Binary front-facing face mask [F].
        camera_face_vertex_uv: Seam-safe per-face UV coordinates [F, 3, 2].

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
        assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert isinstance(
            face_front_facing_mask, torch.Tensor
        ), f"{type(face_front_facing_mask)=}"
        assert isinstance(
            camera_face_vertex_uv, torch.Tensor
        ), f"{type(camera_face_vertex_uv)=}"
        assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
        assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert face_front_facing_mask.shape == (
            faces.shape[0],
        ), f"{face_front_facing_mask.shape=} {faces.shape=}"
        assert camera_face_vertex_uv.shape == (
            faces.shape[0],
            3,
            2,
        ), f"{camera_face_vertex_uv.shape=} {faces.shape=}"

    _validate_inputs()

    vertex_pixels = _geom._camera_vertices_to_pixel(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
    )
    all_face_screen_vertices = vertex_pixels[faces]
    all_face_vertex_depth = vertices_camera[faces][:, :, 2]
    front_face_indices = torch.nonzero(
        face_front_facing_mask,
        as_tuple=False,
    ).reshape(-1)
    if front_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, 3, 2),
                device=vertices_camera.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=vertices_camera.device,
                dtype=torch.long,
            ),
        )

    face_screen_vertices = all_face_screen_vertices[front_face_indices]
    face_vertex_depth = all_face_vertex_depth[front_face_indices]
    face_vertex_uv = camera_face_vertex_uv[front_face_indices]
    (
        visible_screen_polygon_vertices,
        visible_screen_polygon_vertex_counts,
        visible_screen_polygon_face_indices,
    ) = _compute_visible_screen_space_polygon_regions_inside_camera_pixels(
        face_screen_vertices=face_screen_vertices,
        face_vertex_depth=face_vertex_depth,
        image_height=image_height,
        image_width=image_width,
    )
    if visible_screen_polygon_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, visible_screen_polygon_vertices.shape[1], 2),
                device=vertices_camera.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=vertices_camera.device,
                dtype=torch.long,
            ),
        )

    return _map_visible_screen_space_polygon_regions_to_uv(
        visible_screen_polygon_vertices=visible_screen_polygon_vertices,
        visible_screen_polygon_vertex_counts=visible_screen_polygon_vertex_counts,
        visible_screen_polygon_face_indices=visible_screen_polygon_face_indices,
        face_screen_vertices=face_screen_vertices,
        face_vertex_depth=face_vertex_depth,
        face_vertex_uv=face_vertex_uv,
    )


def _compute_visible_screen_space_polygon_regions_inside_camera_pixels(
    face_screen_vertices: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact visible screen-space polygon regions inside each camera pixel.

    Args:
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
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
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert face_screen_vertices.ndim == 3, f"{face_screen_vertices.shape=}"
        assert face_screen_vertices.shape[1:] == (
            3,
            2,
        ), f"{face_screen_vertices.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_vertices.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_vertices.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    (
        clipped_polygon_vertices,
        clipped_polygon_vertex_counts,
        clipped_pixel_indices,
        clipped_face_indices,
    ) = _compute_face_pixel_polygon_intersections_without_occlusion(
        face_screen_vertices=face_screen_vertices,
        image_height=image_height,
        image_width=image_width,
    )
    if clipped_face_indices.numel() == 0:
        return (
            torch.zeros(
                (0, clipped_polygon_vertices.shape[1], 2),
                device=face_screen_vertices.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                device=face_screen_vertices.device,
                dtype=torch.long,
            ),
            torch.zeros(
                (0,),
                device=face_screen_vertices.device,
                dtype=torch.long,
            ),
        )

    return _compute_visible_screen_space_polygon_regions_with_occlusion(
        clipped_polygon_vertices=clipped_polygon_vertices,
        clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
        clipped_pixel_indices=clipped_pixel_indices,
        clipped_face_indices=clipped_face_indices,
        face_screen_vertices=face_screen_vertices,
        face_vertex_depth=face_vertex_depth,
        image_height=image_height,
        image_width=image_width,
    )


def _compute_face_pixel_polygon_intersections_without_occlusion(
    face_screen_vertices: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all face-pixel polygon intersections without considering occlusion.

    Args:
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
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
            face_screen_vertices.shape[0],
            device=face_screen_vertices.device,
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
            Clipped polygon vertices and counts for each candidate pair.
        """
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
        return _geom._clip_convex_polygons_to_pixel_squares(
            polygon_vertices=polygon_vertices,
            polygon_vertex_counts=polygon_vertex_counts,
            pixel_x=pixel_x.to(dtype=torch.float32),
            pixel_y=pixel_y.to(dtype=torch.float32),
        )

    def _pack_valid_face_pixel_polygons(
        clipped_polygon_vertices: torch.Tensor,
        clipped_polygon_vertex_counts: torch.Tensor,
        pixel_x: torch.Tensor,
        pixel_y: torch.Tensor,
        repeated_face_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reject degenerate overlaps and pack the surviving polygons.

        Args:
            clipped_polygon_vertices: Clipped polygon vertices [P, Vmax, 2].
            clipped_polygon_vertex_counts: Clipped polygon vertex counts [P].
            pixel_x: Pixel x per candidate pair [P].
            pixel_y: Pixel y per candidate pair [P].
            repeated_face_indices: Face index per candidate pair [P].

        Returns:
            Packed valid polygons, counts, pixel indices, and face indices.
        """
        clipped_polygon_area = _geom._compute_convex_polygon_areas(
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

    repeated_face_indices, pixel_x, pixel_y = _enumerate_candidate_face_pixel_pairs(
        pair_count_per_face=pair_count_per_face,
        pixel_x_start=pixel_x_start,
        pixel_y_start=pixel_y_start,
        pixel_x_count=pixel_x_count,
        total_pair_count=total_pair_count,
    )
    clipped_polygon_vertices, clipped_polygon_vertex_counts = (
        _clip_face_triangles_to_pixel_squares(
            repeated_face_indices=repeated_face_indices,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            total_pair_count=total_pair_count,
        )
    )
    return _pack_valid_face_pixel_polygons(
        clipped_polygon_vertices=clipped_polygon_vertices,
        clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        repeated_face_indices=repeated_face_indices,
    )


def _compute_visible_screen_space_polygon_regions_with_occlusion(
    clipped_polygon_vertices: torch.Tensor,
    clipped_polygon_vertex_counts: torch.Tensor,
    clipped_pixel_indices: torch.Tensor,
    clipped_face_indices: torch.Tensor,
    face_screen_vertices: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute inter-polygon occlusion and remove hidden screen-space regions.

    Args:
        clipped_polygon_vertices: Face-pixel polygons [P, Vmax, 2].
        clipped_polygon_vertex_counts: Valid polygon vertex counts [P].
        clipped_pixel_indices: Pixel indices [P, 2] in `(y, x)` order.
        clipped_face_indices: Local face indices [P].
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
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
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert clipped_polygon_vertices.ndim == 3, f"{clipped_polygon_vertices.shape=}"
        assert (
            clipped_polygon_vertices.shape[2] == 2
        ), f"{clipped_polygon_vertices.shape=}"
        assert clipped_polygon_vertex_counts.shape == (
            clipped_polygon_vertices.shape[0],
        ), f"{clipped_polygon_vertex_counts.shape=} {clipped_polygon_vertices.shape=}"
        assert clipped_pixel_indices.shape == (
            clipped_polygon_vertices.shape[0],
            2,
        ), f"{clipped_pixel_indices.shape=} {clipped_polygon_vertices.shape=}"
        assert clipped_face_indices.shape == (
            clipped_polygon_vertices.shape[0],
        ), f"{clipped_face_indices.shape=} {clipped_polygon_vertices.shape=}"
        assert face_screen_vertices.ndim == 3, f"{face_screen_vertices.shape=}"
        assert face_screen_vertices.shape[1:] == (
            3,
            2,
        ), f"{face_screen_vertices.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_vertices.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_vertices.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    def _compute_face_inverse_depth_coefficients() -> torch.Tensor:
        """Compute affine inverse-depth coefficients for the projected faces.

        Args:
            None.

        Returns:
            Inverse-depth coefficients [F, 3].
        """
        return _geom._compute_face_inverse_depth_coefficients(
            face_screen_vertices=face_screen_vertices,
            face_vertex_depth=face_vertex_depth,
        )

    def _build_exact_visible_face_pixel_polygons(
        face_inverse_depth_coefficients: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve exact visible face-pixel polygons from clipped overlaps.

        Args:
            face_inverse_depth_coefficients: Inverse-depth coefficients [F, 3].

        Returns:
            Visible polygon vertices, counts, and local face indices.
        """
        return _geom._build_visible_face_pixel_polygons(
            clipped_polygon_vertices=clipped_polygon_vertices,
            clipped_polygon_vertex_counts=clipped_polygon_vertex_counts,
            clipped_pixel_indices=clipped_pixel_indices,
            clipped_face_indices=clipped_face_indices,
            face_inverse_depth_coefficients=face_inverse_depth_coefficients,
        )

    def _pack_visible_polygon_outputs(
        visible_polygon_vertices: torch.Tensor,
        visible_polygon_vertex_counts: torch.Tensor,
        visible_polygon_face_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack exact visible polygons into the downstream tensor format.

        Args:
            visible_polygon_vertices: Visible polygons [N, Vmax, 2].
            visible_polygon_vertex_counts: Visible polygon vertex counts [N].
            visible_polygon_face_indices: Visible local face indices [N].

        Returns:
            Visible polygon vertices, counts, and local face indices.
        """
        return (
            visible_polygon_vertices.contiguous(),
            visible_polygon_vertex_counts.contiguous(),
            visible_polygon_face_indices.contiguous(),
        )

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

    face_inverse_depth_coefficients = _compute_face_inverse_depth_coefficients()
    (
        visible_polygon_vertices,
        visible_polygon_vertex_counts,
        visible_polygon_face_indices,
    ) = _build_exact_visible_face_pixel_polygons(
        face_inverse_depth_coefficients=face_inverse_depth_coefficients,
    )
    return _pack_visible_polygon_outputs(
        visible_polygon_vertices=visible_polygon_vertices,
        visible_polygon_vertex_counts=visible_polygon_vertex_counts,
        visible_polygon_face_indices=visible_polygon_face_indices,
    )


def _map_visible_screen_space_polygon_regions_to_uv(
    visible_screen_polygon_vertices: torch.Tensor,
    visible_screen_polygon_vertex_counts: torch.Tensor,
    visible_screen_polygon_face_indices: torch.Tensor,
    face_screen_vertices: torch.Tensor,
    face_vertex_depth: torch.Tensor,
    face_vertex_uv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map visible screen-space polygon regions into UV.

    Args:
        visible_screen_polygon_vertices: Visible screen polygons [N, Vmax, 2].
        visible_screen_polygon_vertex_counts: Valid polygon vertex counts [N].
        visible_screen_polygon_face_indices: Local face index for each visible polygon [N].
        face_screen_vertices: Projected triangle vertices [F, 3, 2].
        face_vertex_depth: Camera-space vertex depths [F, 3].
        face_vertex_uv: Seam-safe per-face UV coordinates [F, 3, 2].

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
            visible_screen_polygon_vertices, torch.Tensor
        ), f"{type(visible_screen_polygon_vertices)=}"
        assert isinstance(
            visible_screen_polygon_vertex_counts, torch.Tensor
        ), f"{type(visible_screen_polygon_vertex_counts)=}"
        assert isinstance(
            visible_screen_polygon_face_indices, torch.Tensor
        ), f"{type(visible_screen_polygon_face_indices)=}"
        assert isinstance(
            face_screen_vertices, torch.Tensor
        ), f"{type(face_screen_vertices)=}"
        assert isinstance(
            face_vertex_depth, torch.Tensor
        ), f"{type(face_vertex_depth)=}"
        assert isinstance(face_vertex_uv, torch.Tensor), f"{type(face_vertex_uv)=}"
        assert (
            visible_screen_polygon_vertices.ndim == 3
        ), f"{visible_screen_polygon_vertices.shape=}"
        assert (
            visible_screen_polygon_vertices.shape[2] == 2
        ), f"{visible_screen_polygon_vertices.shape=}"
        assert visible_screen_polygon_vertex_counts.shape == (
            visible_screen_polygon_vertices.shape[0],
        ), f"{visible_screen_polygon_vertex_counts.shape=} {visible_screen_polygon_vertices.shape=}"
        assert visible_screen_polygon_face_indices.shape == (
            visible_screen_polygon_vertices.shape[0],
        ), f"{visible_screen_polygon_face_indices.shape=} {visible_screen_polygon_vertices.shape=}"
        assert face_screen_vertices.ndim == 3, f"{face_screen_vertices.shape=}"
        assert face_screen_vertices.shape[1:] == (
            3,
            2,
        ), f"{face_screen_vertices.shape=}"
        assert face_vertex_depth.shape == (
            face_screen_vertices.shape[0],
            3,
        ), f"{face_vertex_depth.shape=} {face_screen_vertices.shape=}"
        assert face_vertex_uv.shape == (
            face_screen_vertices.shape[0],
            3,
            2,
        ), f"{face_vertex_uv.shape=} {face_screen_vertices.shape=}"

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
            Screen-space vertices, depths, and UVs for each visible polygon.
        """
        return (
            face_screen_vertices[visible_screen_polygon_face_indices].contiguous(),
            face_vertex_depth[visible_screen_polygon_face_indices].contiguous(),
            face_vertex_uv[visible_screen_polygon_face_indices].contiguous(),
        )

    def _project_screen_polygon_vertices_to_uv(
        polygon_face_screen_vertices: torch.Tensor,
        polygon_face_vertex_depth: torch.Tensor,
        polygon_face_vertex_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Project visible screen polygons into UV.

        Args:
            polygon_face_screen_vertices: Owning face screen vertices [N, 3, 2].
            polygon_face_vertex_depth: Owning face depths [N, 3].
            polygon_face_vertex_uv: Owning face UVs [N, 3, 2].

        Returns:
            Visible UV polygons [N, Vmax, 2].
        """
        return _geom._project_screen_polygons_to_face_uv(
            polygon_vertices=visible_screen_polygon_vertices,
            face_screen_vertices=polygon_face_screen_vertices,
            face_vertex_depth=polygon_face_vertex_depth,
            face_vertex_uv=polygon_face_vertex_uv,
        )

    def _pack_visible_uv_polygons(
        uv_polygon_vertices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack UV polygons with their original vertex counts.

        Args:
            uv_polygon_vertices: Visible UV polygons [N, Vmax, 2].

        Returns:
            UV polygons and their vertex counts.
        """
        return (
            uv_polygon_vertices.contiguous(),
            visible_screen_polygon_vertex_counts.contiguous(),
        )

    (
        polygon_face_screen_vertices,
        polygon_face_vertex_depth,
        polygon_face_vertex_uv,
    ) = _gather_visible_polygon_face_geometry()
    uv_polygon_vertices = _project_screen_polygon_vertices_to_uv(
        polygon_face_screen_vertices=polygon_face_screen_vertices,
        polygon_face_vertex_depth=polygon_face_vertex_depth,
        polygon_face_vertex_uv=polygon_face_vertex_uv,
    )
    return _pack_visible_uv_polygons(
        uv_polygon_vertices=uv_polygon_vertices,
    )


def _compute_visible_uv_texels_from_uv_polygon_regions(
    uv_polygon_vertices: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
    polygon_rast_method: str = "v2",
) -> torch.Tensor:
    """Compute visible UV texels from the UV polygon regions.

    Args:
        uv_polygon_vertices: Visible UV polygons [N, Vmax, 2].
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
            uv_polygon_vertices, torch.Tensor
        ), f"{type(uv_polygon_vertices)=}"
        assert isinstance(
            uv_polygon_vertex_counts, torch.Tensor
        ), f"{type(uv_polygon_vertex_counts)=}"
        assert isinstance(texture_size, int), f"{type(texture_size)=}"
        assert isinstance(polygon_rast_method, str), f"{type(polygon_rast_method)=}"
        assert uv_polygon_vertices.ndim == 3, f"{uv_polygon_vertices.shape=}"
        assert uv_polygon_vertices.shape[2] == 2, f"{uv_polygon_vertices.shape=}"
        assert uv_polygon_vertex_counts.shape == (
            uv_polygon_vertices.shape[0],
        ), f"{uv_polygon_vertex_counts.shape=} {uv_polygon_vertices.shape=}"
        assert texture_size > 0, f"{texture_size=}"
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )

    _validate_inputs()

    if polygon_rast_method == "v1":
        covered_texel_indices = _compute_uv_polygon_texel_contributions_v1(
            uv_polygon_vertices=uv_polygon_vertices,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=texture_size,
        )
    else:
        covered_texel_indices = _compute_uv_polygon_texel_contributions_v2(
            uv_polygon_vertices=uv_polygon_vertices,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
            texture_size=texture_size,
        )
    uv_mask = torch.zeros(
        (1, texture_size, texture_size, 1),
        device=uv_polygon_vertices.device,
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
    uv_polygon_vertices: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Construct exact step-2 `v1` texel contributions for visible UV polygons.

    Args:
        uv_polygon_vertices: Visible UV polygons [N, Vmax, 2].
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
            uv_polygon_vertices, torch.Tensor
        ), f"{type(uv_polygon_vertices)=}"
        assert isinstance(
            uv_polygon_vertex_counts, torch.Tensor
        ), f"{type(uv_polygon_vertex_counts)=}"
        assert uv_polygon_vertices.ndim == 3, f"{uv_polygon_vertices.shape=}"
        assert uv_polygon_vertices.shape[2] == 2, f"{uv_polygon_vertices.shape=}"
        assert uv_polygon_vertex_counts.shape == (
            uv_polygon_vertices.shape[0],
        ), f"{uv_polygon_vertex_counts.shape=} {uv_polygon_vertices.shape=}"
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
        return _geom._duplicate_wrapped_uv_polygons(
            uv_polygon_vertices=uv_polygon_vertices,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
        )

    (
        wrapped_uv_polygon_vertices,
        wrapped_uv_polygon_vertex_counts,
    ) = _duplicate_wrap_crossing_polygons()
    return _geom._build_uv_polygon_texel_intersections(
        uv_polygon_vertices=wrapped_uv_polygon_vertices,
        uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
        texture_size=texture_size,
    )


def _compute_uv_polygon_texel_contributions_v2(
    uv_polygon_vertices: torch.Tensor,
    uv_polygon_vertex_counts: torch.Tensor,
    texture_size: int,
) -> torch.Tensor:
    """Construct approximate step-2 `v2` texel contributions for visible UV polygons.

    Args:
        uv_polygon_vertices: Visible UV polygons [N, Vmax, 2].
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
            "Expected `uv_polygon_vertex_counts` to align with polygon count. "
            f"{uv_polygon_vertex_counts.shape=} {uv_polygon_vertices.shape=}."
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
        return _geom._duplicate_wrapped_uv_polygons(
            uv_polygon_vertices=uv_polygon_vertices,
            uv_polygon_vertex_counts=uv_polygon_vertex_counts,
        )

    def _triangulate_wrapped_uv_polygons(
        wrapped_uv_polygon_vertices: torch.Tensor,
        wrapped_uv_polygon_vertex_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Triangulate wrapped convex UV polygons into a triangle soup.

        Args:
            wrapped_uv_polygon_vertices: Wrapped UV polygons [Nw, Vmax, 2].
            wrapped_uv_polygon_vertex_counts: Wrapped UV polygon vertex counts [Nw].

        Returns:
            Wrapped UV triangle soup [K, 3, 2].
        """

        def _validate_wrapped_inputs() -> None:
            """Validate input arguments.

            Args:
                None.

            Returns:
                None.
            """
            assert isinstance(wrapped_uv_polygon_vertices, torch.Tensor), (
                "Expected `wrapped_uv_polygon_vertices` to be a tensor. "
                f"Got {type(wrapped_uv_polygon_vertices)=}."
            )
            assert isinstance(wrapped_uv_polygon_vertex_counts, torch.Tensor), (
                "Expected `wrapped_uv_polygon_vertex_counts` to be a tensor. "
                f"Got {type(wrapped_uv_polygon_vertex_counts)=}."
            )
            assert wrapped_uv_polygon_vertices.ndim == 3, (
                "Expected `wrapped_uv_polygon_vertices` to be rank-3. "
                f"{wrapped_uv_polygon_vertices.shape=}."
            )
            assert wrapped_uv_polygon_vertices.shape[2] == 2, (
                "Expected `wrapped_uv_polygon_vertices` to end with UV pairs. "
                f"{wrapped_uv_polygon_vertices.shape=}."
            )
            assert wrapped_uv_polygon_vertex_counts.shape == (
                wrapped_uv_polygon_vertices.shape[0],
            ), (
                "Expected wrapped vertex counts to align with polygon count. "
                f"{wrapped_uv_polygon_vertex_counts.shape=} "
                f"{wrapped_uv_polygon_vertices.shape=}."
            )

        _validate_wrapped_inputs()

        return _geom._triangulate_convex_uv_polygons(
            polygon_vertices=wrapped_uv_polygon_vertices,
            polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
        )

    (
        wrapped_uv_polygon_vertices,
        wrapped_uv_polygon_vertex_counts,
    ) = _duplicate_wrap_crossing_polygons()
    wrapped_uv_triangles = _triangulate_wrapped_uv_polygons(
        wrapped_uv_polygon_vertices=wrapped_uv_polygon_vertices,
        wrapped_uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts,
    )
    return _geom._build_uv_triangle_texel_intersections_v2(
        uv_triangles=wrapped_uv_triangles,
        texture_size=texture_size,
    )
