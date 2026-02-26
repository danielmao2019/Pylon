"""Texture extraction utilities for generic triangle meshes."""

from typing import Dict, List, Optional, Tuple, Union

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.point_cloud.camera.transform import (
    world_to_camera_transform,
)
from models.three_d.meshes.ops.normals import compute_vertex_normals


def extract_texture_from_images(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    images: Union[torch.Tensor, List[torch.Tensor]],
    cameras: Cameras,
    representation: str = "vertex_color",
    weights: str = "visible",
    vertex_uv: Optional[torch.Tensor] = None,
    texture_size: int = 1024,
    default_color: float = 0.7,
    return_valid_mask: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract texture from multi-view RGB images.

    Args:
        vertices: Mesh vertices in world coordinates with shape [V, 3].
        faces: Mesh faces with shape [F, 3], required for UV extraction.
        images: Multi-view RGB images as [N, 3, H, W] or [N, H, W, 3], or list of [3, H, W].
        cameras: Per-view cameras in OpenCV convention.
        representation: Output texture representation, "vertex_color" or "uv_texture_map".
        weights: Per-view fusion weighting mode, "visible" or "normals".
        vertex_uv: Per-vertex UV coordinates [V, 2], required for UV extraction.
        texture_size: UV texture resolution for UV extraction.
        default_color: Fallback color for texels/vertices without any valid observation.
        return_valid_mask: Whether to also return a binary valid-observation mask.

    Returns:
        If return_valid_mask is False:
            Texture tensor in selected representation:
            [V, 3] for vertex colors, or [1, texture_size, texture_size, 3] for UV texture map.
        If return_valid_mask is True:
            Dict containing:
                "texture": texture tensor in selected representation.
                "valid_mask": binary mask of valid observations:
                    [V, 1] for vertex colors, or [1, texture_size, texture_size, 1] for UV texture map.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert faces is None or isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(images, torch.Tensor) or isinstance(
        images, list
    ), f"{type(images)=}"
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(representation, str), f"{type(representation)=}"
    assert representation in ("vertex_color", "uv_texture_map"), f"{representation=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert vertex_uv is None or isinstance(
        vertex_uv, torch.Tensor
    ), f"{type(vertex_uv)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert isinstance(return_valid_mask, bool), f"{type(return_valid_mask)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert not isinstance(images, torch.Tensor) or images.ndim == 4
    assert not isinstance(images, list) or len(images) > 0
    assert not isinstance(images, list) or all(
        isinstance(image, torch.Tensor) for image in images
    )
    assert faces is not None or weights == "visible"
    assert faces is not None or representation == "vertex_color"
    assert vertex_uv is not None or representation == "vertex_color"
    assert faces is None or (
        faces.ndim == 2
        and faces.shape[1] == 3
        and faces.dtype in (torch.int32, torch.int64)
    )
    assert vertex_uv is None or (
        vertex_uv.ndim == 2
        and vertex_uv.shape[1] == 2
        and vertex_uv.shape[0] == vertices.shape[0]
    )

    # Input normalizations
    if isinstance(images, list):
        image_stack = torch.stack(images, dim=0)
        image_count = len(images)
    else:
        image_stack = images
        image_count = image_stack.shape[0]
    assert len(cameras) == image_count, f"{len(cameras)=} {image_count=}"

    if image_stack.shape[1] == 3:
        images_nchw = image_stack
    else:
        assert image_stack.shape[3] == 3, f"{image_stack.shape=}"
        images_nchw = image_stack.permute(0, 3, 1, 2).contiguous()

    if images_nchw.dtype == torch.uint8:
        images_nchw = images_nchw.to(dtype=torch.float32) / 255.0
    else:
        images_nchw = images_nchw.to(dtype=torch.float32)

    device = vertices.device
    vertices = vertices.to(device=device, dtype=torch.float32)
    images_nchw = images_nchw.to(device=device)
    cameras = cameras.to(device=device, convention="opencv")

    if representation == "vertex_color":
        extracted_vertex_color = _extract_vertex_color_from_images(
            vertices=vertices,
            faces=faces,
            images_nchw=images_nchw,
            cameras=cameras,
            weights=weights,
            default_color=default_color,
        )
        if not return_valid_mask:
            return extracted_vertex_color["texture"]
        return extracted_vertex_color

    assert representation == "uv_texture_map", f"{representation=}"
    assert faces is not None, f"{faces=}"
    assert vertex_uv is not None, f"{vertex_uv=}"
    extracted_uv_texture_map = _extract_uv_texture_map_from_images(
        vertices=vertices,
        faces=faces,
        images_nchw=images_nchw,
        cameras=cameras,
        weights=weights,
        vertex_uv=vertex_uv,
        texture_size=texture_size,
        default_color=default_color,
    )
    if not return_valid_mask:
        return extracted_uv_texture_map["texture"]
    return extracted_uv_texture_map


def _extract_vertex_color_from_images(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    images_nchw: torch.Tensor,
    cameras: Cameras,
    weights: str,
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Fuse per-view projected vertex colors into one vertex-color tensor.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Optional mesh faces [F, 3], needed when weights == "normals".
        images_nchw: Input RGB images [N, 3, H, W].
        cameras: Per-view cameras.
        weights: Fusion weighting mode, "visible" or "normals".
        default_color: Fallback color for vertices without valid observations.

    Returns:
        Dict with:
            "texture": fused vertex colors [V, 3].
            "valid_mask": binary valid-observation mask [V, 1].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert faces is None or isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(images_nchw, torch.Tensor), f"{type(images_nchw)=}"
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert images_nchw.ndim == 4, f"{images_nchw.shape=}"
    assert images_nchw.shape[1] == 3, f"{images_nchw.shape=}"
    assert len(cameras) == images_nchw.shape[0], f"{len(cameras)=} {images_nchw.shape=}"

    device = vertices.device
    vertex_count = vertices.shape[0]
    color_numerator = torch.zeros((vertex_count, 3), device=device, dtype=torch.float32)
    weight_denominator = torch.zeros(
        (vertex_count, 1), device=device, dtype=torch.float32
    )

    for view_idx in range(images_nchw.shape[0]):
        extracted_single_image = _extract_vertex_color_from_single_image(
            vertices=vertices,
            faces=faces,
            image=images_nchw[view_idx],
            camera=cameras[view_idx : view_idx + 1],
            weights=weights,
            default_color=default_color,
        )
        texture = extracted_single_image["texture"]
        weight = extracted_single_image["weight"]
        color_numerator = color_numerator + texture * weight
        weight_denominator = weight_denominator + weight

    vertex_color = torch.full(
        (vertex_count, 3),
        fill_value=default_color,
        device=device,
        dtype=torch.float32,
    )
    has_weight = weight_denominator > 0.0
    vertex_color = torch.where(
        has_weight.expand_as(vertex_color),
        color_numerator / (weight_denominator + 1e-6),
        vertex_color,
    )
    return {
        "texture": vertex_color.clamp(0.0, 1.0).contiguous(),
        "valid_mask": has_weight.to(dtype=torch.float32).contiguous(),
    }


def _extract_vertex_color_from_single_image(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    image: torch.Tensor,
    camera: Cameras,
    weights: str,
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Extract one-view vertex colors and corresponding per-vertex weights.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Optional mesh faces [F, 3], needed when weights == "normals".
        image: One RGB image [3, H, W].
        camera: One camera instance.
        weights: Weighting mode, "visible" or "normals".
        default_color: Fallback color for invalid projections.

    Returns:
        Dict with:
            "texture": Projected vertex RGB colors [V, 3].
            "weight": Per-vertex weights [V, 1].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert faces is None or isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert image.ndim == 3, f"{image.shape=}"
    assert image.shape[0] == 3, f"{image.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    visibility_mask = _compute_v_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    if weights == "normals":
        assert faces is not None, f"{faces=}"
        normals_weight = _compute_v_normals_weights(
            vertices=vertices,
            faces=faces.to(device=vertices.device, dtype=torch.long),
            camera=camera,
        )
        vertex_weight = visibility_mask * normals_weight
    else:
        vertex_weight = visibility_mask

    vertex_color = _project_v_colors(
        vertices=vertices,
        image=image,
        camera=camera,
        default_color=default_color,
    )
    return {
        "texture": vertex_color,
        "weight": vertex_weight.unsqueeze(1),
    }


def _compute_v_visibility_mask(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute one-view binary visibility mask over vertices.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Optional mesh faces [F, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Float tensor [V] with values in {0, 1}.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert faces is None or isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"

    _xy, _depth, vertices_camera, projection_valid = _project_vertices_to_image(
        vertices=vertices,
        camera=camera,
        image_height=image_height,
        image_width=image_width,
    )
    if faces is not None:
        visible_vertex_mask = _compute_rasterized_visible_vertex_mask(
            vertices_camera=vertices_camera,
            faces=faces.to(device=vertices.device, dtype=torch.long).contiguous(),
            intrinsics=camera[0].intrinsics,
            image_height=image_height,
            image_width=image_width,
        )
        visibility_bool = projection_valid & visible_vertex_mask
    else:
        visibility_bool = projection_valid

    return visibility_bool.to(dtype=torch.float32)


def _compute_v_normals_weights(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Compute one-view per-vertex normal-alignment weights.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        camera: One camera instance.

    Returns:
        Per-vertex weights [V], computed as max(dot(n, view_dir), 0).
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert faces.dtype == torch.long, f"{faces.dtype=}"
    assert len(camera) == 1, f"{len(camera)=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    normals_camera = compute_vertex_normals(
        base_vertices=vertices_camera,
        faces=faces,
    ).to(device=vertices.device, dtype=torch.float32)
    normals_camera_norm = torch.linalg.norm(normals_camera, dim=1)
    normals_camera_norm_error = torch.max(torch.abs(normals_camera_norm - 1.0))
    assert (
        float(normals_camera_norm_error) <= 1.0e-5
    ), f"{float(normals_camera_norm_error)=}"

    view_direction = F.normalize(-vertices_camera, p=2, dim=1)
    alignment = (normals_camera * view_direction).sum(dim=1).clamp(0.0, 1.0)
    return alignment


def _project_v_colors(
    vertices: torch.Tensor,
    image: torch.Tensor,
    camera: Cameras,
    default_color: float,
) -> torch.Tensor:
    """Project one image to vertices and sample per-vertex RGB colors.

    Args:
        vertices: Mesh vertices [V, 3].
        image: One RGB image [3, H, W].
        camera: One camera instance.
        default_color: Fallback color for invalid projections.

    Returns:
        Vertex RGB colors with shape [V, 3].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert image.ndim == 3, f"{image.shape=}"
    assert image.shape[0] == 3, f"{image.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    xy, _depth, _vertices_camera, projection_valid = _project_vertices_to_image(
        vertices=vertices,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )

    vertex_count = vertices.shape[0]
    vertex_color = torch.full(
        (vertex_count, 3),
        fill_value=default_color,
        device=image.device,
        dtype=torch.float32,
    )
    if torch.any(projection_valid):
        x_idx = torch.round(xy[:, 0]).to(dtype=torch.long)
        y_idx = torch.round(xy[:, 1]).to(dtype=torch.long)
        valid_indices = torch.nonzero(projection_valid, as_tuple=False).reshape(-1)
        sampled_color = image[
            :, y_idx[projection_valid], x_idx[projection_valid]
        ].transpose(0, 1)
        vertex_color[valid_indices] = sampled_color
    return vertex_color.contiguous()


def _extract_uv_texture_map_from_images(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    images_nchw: torch.Tensor,
    cameras: Cameras,
    weights: str,
    vertex_uv: torch.Tensor,
    texture_size: int,
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Fuse per-view UV observations into one UV texture map.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        images_nchw: Input RGB images [N, 3, H, W].
        cameras: Per-view cameras.
        weights: Fusion weighting mode, "visible" or "normals".
        vertex_uv: Per-vertex UV coordinates [V, 2].
        texture_size: UV texture resolution.
        default_color: Fallback color for UV pixels without valid observations.

    Returns:
        Dict with:
            "texture": fused UV texture map [1, T, T, 3].
            "valid_mask": binary valid-observation mask [1, T, T, 1].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(images_nchw, torch.Tensor), f"{type(images_nchw)=}"
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert images_nchw.ndim == 4, f"{images_nchw.shape=}"
    assert images_nchw.shape[1] == 3, f"{images_nchw.shape=}"
    assert len(cameras) == images_nchw.shape[0], f"{len(cameras)=} {images_nchw.shape=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert (
        vertex_uv.shape[0] == vertices.shape[0]
    ), f"{vertex_uv.shape=} {vertices.shape=}"

    device = vertices.device
    faces_long = faces.to(device=device, dtype=torch.long).contiguous()
    uv_rasterization_data = _build_uv_rasterization_data(
        vertices=vertices,
        vertex_uv=vertex_uv.to(device=device, dtype=torch.float32),
        faces=faces_long,
        texture_size=texture_size,
    )

    uv_numerator = torch.zeros(
        (1, texture_size, texture_size, 3), device=device, dtype=torch.float32
    )
    uv_denominator = torch.zeros(
        (1, texture_size, texture_size, 1), device=device, dtype=torch.float32
    )

    for view_idx in range(images_nchw.shape[0]):
        extracted_single_image = _extract_uv_texture_map_from_single_image(
            vertices=vertices,
            faces=faces_long,
            image=images_nchw[view_idx],
            camera=cameras[view_idx : view_idx + 1],
            weights=weights,
            uv_rasterization_data=uv_rasterization_data,
        )
        texture = extracted_single_image["texture"]
        weight = extracted_single_image["weight"]
        uv_numerator = uv_numerator + texture * weight
        uv_denominator = uv_denominator + weight

    uv_texture_map = torch.full(
        (1, texture_size, texture_size, 3),
        fill_value=default_color,
        device=device,
        dtype=torch.float32,
    )
    has_weight = uv_denominator > 0.0
    uv_texture_map = torch.where(
        has_weight.expand_as(uv_texture_map),
        uv_numerator / (uv_denominator + 1e-6),
        uv_texture_map,
    )
    return {
        "texture": uv_texture_map.clamp(0.0, 1.0).contiguous(),
        "valid_mask": has_weight.to(dtype=torch.float32).contiguous(),
    }


def _extract_uv_texture_map_from_single_image(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image: torch.Tensor,
    camera: Cameras,
    weights: str,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract one-view UV texture observation and UV weight map.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        image: One RGB image [3, H, W].
        camera: One camera instance.
        weights: Weighting mode, "visible" or "normals".
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        Dict with:
            "texture": UV RGB image [1, T, T, 3] from this view.
            "weight": UV weight map [1, T, T, 1] from this view.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert isinstance(uv_rasterization_data, dict), f"{type(uv_rasterization_data)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert faces.dtype == torch.long, f"{faces.dtype=}"
    assert image.ndim == 3, f"{image.shape=}"
    assert image.shape[0] == 3, f"{image.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    uv_visibility_mask = _compute_f_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
        uv_rasterization_data=uv_rasterization_data,
    )
    if weights == "normals":
        face_normals_weight = _compute_f_normals_weights(
            vertices=vertices,
            faces=faces,
            camera=camera,
        )
        uv_normals_weight = _rasterize_face_weights_to_uv(
            face_weight=face_normals_weight,
            uv_rasterization_data=uv_rasterization_data,
        )
        uv_weight = uv_visibility_mask * uv_normals_weight
    else:
        uv_weight = uv_visibility_mask

    uv_texture = _project_f_colors(
        vertices=vertices,
        image=image,
        camera=camera,
        uv_rasterization_data=uv_rasterization_data,
    )
    return {
        "texture": uv_texture,
        "weight": uv_weight,
    }


def _compute_f_visibility_mask(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute one-view UV-pixel visibility mask from camera z-buffer consistency.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        Float tensor [1, T, T, 1] with values in {0, 1}.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert isinstance(uv_rasterization_data, dict), f"{type(uv_rasterization_data)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert faces.dtype == torch.long, f"{faces.dtype=}"
    assert len(camera) == 1, f"{len(camera)=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"
    assert "tri_i32" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert "rast_out" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert "uv_mask" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    tri_i32 = uv_rasterization_data["tri_i32"]
    rast_out = uv_rasterization_data["rast_out"]
    uv_mask = uv_rasterization_data["uv_mask"]
    assert isinstance(tri_i32, torch.Tensor), f"{type(tri_i32)=}"
    assert isinstance(rast_out, torch.Tensor), f"{type(rast_out)=}"
    assert isinstance(uv_mask, torch.Tensor), f"{type(uv_mask)=}"

    # === Criterion 1: UV texel belongs to a valid mesh face ===
    uv_vertices_camera, _ = dr.interpolate(
        attr=vertices_camera.unsqueeze(0).contiguous(),
        rast=rast_out,
        tri=tri_i32,
    )
    uv_face_plus1 = rast_out[..., 3:4].to(dtype=torch.long)
    uv_has_face = uv_face_plus1 > 0

    # === Criterion 2: Texel 3D point projects to a valid image pixel ===
    intrinsics = camera[0].intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    uv_x_camera = uv_vertices_camera[..., 0]
    uv_y_camera = uv_vertices_camera[..., 1]
    uv_z_camera = uv_vertices_camera[..., 2]
    uv_x_pixel = fx * (uv_x_camera / uv_z_camera) + cx
    uv_y_pixel = fy * (uv_y_camera / uv_z_camera) + cy
    uv_depth_valid = uv_z_camera > 1e-8
    uv_in_bounds = (
        (uv_x_pixel >= 0.0)
        & (uv_x_pixel <= float(image_width - 1))
        & (uv_y_pixel >= 0.0)
        & (uv_y_pixel <= float(image_height - 1))
    )
    uv_projection_valid = uv_depth_valid & uv_in_bounds

    # === Criterion 3: Camera z-buffer front face matches UV texel face ===
    grid_x = uv_x_pixel / float(max(image_width - 1, 1)) * 2.0 - 1.0
    # _render_camera_face_index_buffer stores raster rows bottom-first (OpenGL style),
    # while grid_sample interprets rows top-first; convert projected y accordingly.
    grid_y_for_camera_raster = 1.0 - uv_y_pixel / float(max(image_height - 1, 1)) * 2.0
    sampling_grid = torch.stack(
        [grid_x, grid_y_for_camera_raster],
        dim=-1,
    ).contiguous()

    camera_face_plus1 = _render_camera_face_index_buffer(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).permute(0, 3, 1, 2)
    sampled_face_plus1 = F.grid_sample(
        input=camera_face_plus1,
        grid=sampling_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    sampled_face_plus1 = sampled_face_plus1.to(dtype=torch.long)

    uv_visible = (
        uv_has_face
        & uv_projection_valid.unsqueeze(-1)
        & (sampled_face_plus1 == uv_face_plus1)
    )

    # === Criterion 4: The corresponding face is front-facing to the camera ===
    face_front_facing_mask = (
        _compute_f_normals_weights(
            vertices=vertices,
            faces=faces,
            camera=camera,
        )
        > 0.0
    ).to(dtype=torch.float32)
    uv_front_facing_mask = _rasterize_face_weights_to_uv(
        face_weight=face_front_facing_mask,
        uv_rasterization_data=uv_rasterization_data,
    )
    uv_visible = uv_visible & (uv_front_facing_mask > 0.5)
    return (uv_visible.to(dtype=torch.float32) * uv_mask).contiguous()


def _compute_f_normals_weights(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Compute one-view per-face normal-alignment weights.

    Args:
        vertices: Mesh vertices [V, 3].
        faces: Mesh faces [F, 3].
        camera: One camera instance.

    Returns:
        Per-face weights [F], computed as max(dot(n, view_dir), 0).
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert faces.dtype == torch.long, f"{faces.dtype=}"
    assert len(camera) == 1, f"{len(camera)=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )

    v0_camera = vertices_camera[faces[:, 0]]
    v1_camera = vertices_camera[faces[:, 1]]
    v2_camera = vertices_camera[faces[:, 2]]
    face_normals_camera = torch.cross(
        v1_camera - v0_camera,
        v2_camera - v0_camera,
        dim=1,
    )
    face_normals_camera_norm = torch.linalg.norm(
        face_normals_camera, dim=1, keepdim=True
    )
    assert torch.all(face_normals_camera_norm > 0), f"{face_normals_camera_norm.min()=}"
    face_normals_camera = face_normals_camera / face_normals_camera_norm

    face_centers_camera = (v0_camera + v1_camera + v2_camera) / 3.0
    face_view_direction = F.normalize(-face_centers_camera, p=2, dim=1)
    alignment = (face_normals_camera * face_view_direction).sum(dim=1).clamp(0.0, 1.0)
    return alignment


def _project_f_colors(
    vertices: torch.Tensor,
    image: torch.Tensor,
    camera: Cameras,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Project one image into UV space using rasterized UV correspondence.

    Args:
        vertices: Mesh vertices [V, 3].
        image: One RGB image [3, H, W].
        camera: One camera instance.
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        One-view UV RGB image with shape [1, T, T, 3].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(image, torch.Tensor), f"{type(image)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(uv_rasterization_data, dict), f"{type(uv_rasterization_data)=}"
    assert "tri_i32" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert "rast_out" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert image.ndim == 3, f"{image.shape=}"
    assert image.shape[0] == 3, f"{image.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    xy, _depth, _vertices_camera, _valid = _project_vertices_to_image(
        vertices=vertices,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    tri_i32 = uv_rasterization_data["tri_i32"]
    rast_out = uv_rasterization_data["rast_out"]
    uv_xy, _ = dr.interpolate(
        attr=xy.unsqueeze(0).contiguous(),
        rast=rast_out,
        tri=tri_i32,
    )

    image_height = int(image.shape[1])
    image_width = int(image.shape[2])
    grid_x = uv_xy[..., 0] / float(max(image_width - 1, 1)) * 2.0 - 1.0
    grid_y = uv_xy[..., 1] / float(max(image_height - 1, 1)) * 2.0 - 1.0
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1).contiguous()
    sampled_image = F.grid_sample(
        input=image.unsqueeze(0),
        grid=sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled_image.permute(0, 2, 3, 1).contiguous()


# -----------------------------------------------------------------------------
# Other helpers
# -----------------------------------------------------------------------------


def _build_uv_rasterization_data(
    vertices: torch.Tensor,
    vertex_uv: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int,
) -> Dict[str, torch.Tensor]:
    """Build reusable UV rasterization tensors for UV-space operations.

    Args:
        vertices: Mesh vertices [V, 3].
        vertex_uv: Per-vertex UV coordinates [V, 2].
        faces: Mesh faces [F, 3].
        texture_size: UV texture resolution.

    Returns:
        Dict containing:
            "tri_i32": Face indices as int32 for nvdiffrast.
            "rast_out": UV rasterization output [1, T, T, 4].
            "uv_mask": UV occupancy mask [1, T, T, 1].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert (
        vertex_uv.shape[0] == vertices.shape[0]
    ), f"{vertex_uv.shape=} {vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"

    tri_i32 = faces.to(device=vertex_uv.device, dtype=torch.int32).contiguous()
    overlap_depth_priority = (-vertices[:, 2]).to(
        device=vertex_uv.device,
        dtype=torch.float32,
    )
    uv_clip = _vertex_uv_to_clip(
        vertex_uv=vertex_uv,
        overlap_depth_priority=overlap_depth_priority,
    ).to(device=vertex_uv.device, dtype=torch.float32)
    uv_ctx = dr.RasterizeCudaContext(device=vertex_uv.device)
    rast_out, _ = dr.rasterize(
        glctx=uv_ctx,
        pos=uv_clip.contiguous(),
        tri=tri_i32,
        resolution=[texture_size, texture_size],
        ranges=None,
    )
    uv_mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)

    return {
        "tri_i32": tri_i32,
        "rast_out": rast_out,
        "uv_mask": uv_mask,
    }


def _rasterize_face_weights_to_uv(
    face_weight: torch.Tensor,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Map per-face weights to per-UV-pixel weights for one view.

    Args:
        face_weight: Per-face weights [F].
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        UV weight map [1, T, T, 1].
    """
    # Input validations
    assert isinstance(face_weight, torch.Tensor), f"{type(face_weight)=}"
    assert isinstance(uv_rasterization_data, dict), f"{type(uv_rasterization_data)=}"
    assert "rast_out" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert "uv_mask" in uv_rasterization_data, f"{uv_rasterization_data.keys()=}"
    assert face_weight.ndim == 1, f"{face_weight.shape=}"

    rast_out = uv_rasterization_data["rast_out"]
    uv_mask = uv_rasterization_data["uv_mask"]
    assert isinstance(rast_out, torch.Tensor), f"{type(rast_out)=}"
    assert isinstance(uv_mask, torch.Tensor), f"{type(uv_mask)=}"

    uv_face_indices = rast_out[..., 3].to(dtype=torch.long) - 1
    uv_visible = uv_face_indices >= 0
    uv_weight = torch.zeros_like(uv_mask)
    if torch.any(uv_visible):
        uv_weight_values = face_weight[uv_face_indices[uv_visible]]
        uv_weight[uv_visible.unsqueeze(-1)] = uv_weight_values.reshape(-1)
    uv_weight = uv_weight.clamp(min=0.0) * uv_mask
    return uv_weight


def _render_camera_face_index_buffer(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Render a one-view camera-space face-index buffer.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Face-index image [1, H, W, 1] with values face_index + 1 and 0 for background.
    """
    # Input validations
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
    assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"

    clip_vertices = _camera_vertices_to_clip(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=vertices_camera.device, dtype=torch.float32)
    tri_i32 = faces.to(device=vertices_camera.device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=vertices_camera.device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_vertices.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )

    face_indices = rast_out[..., 3].to(dtype=torch.long) - 1
    face_plus1 = (face_indices + 1).to(dtype=torch.float32).unsqueeze(-1)
    visible = face_indices >= 0
    return torch.where(visible.unsqueeze(-1), face_plus1, torch.zeros_like(face_plus1))


def _vertices_world_to_camera(
    vertices: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Transform one-view world-space vertices to camera-space vertices.

    Args:
        vertices: Mesh vertices in world coordinates [V, 3].
        camera: One camera instance.

    Returns:
        Camera-space vertices [V, 3].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    camera_single = camera[0].to(device=vertices.device, convention="opencv")
    vertices_camera = world_to_camera_transform(
        points=vertices,
        extrinsics=camera_single.extrinsics,
        inplace=False,
    )
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert (
        vertices_camera.shape == vertices.shape
    ), f"{vertices_camera.shape=} {vertices.shape=}"
    return vertices_camera


def _project_vertices_to_image(
    vertices: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space vertices to image pixels for one view.

    Args:
        vertices: Mesh vertices in world coordinates [V, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        A tuple of:
            xy: Pixel coordinates [V, 2].
            depth: Camera-space depth [V].
            vertices_camera: Camera-space vertices [V, 3].
            valid: In-frame projection validity mask [V].
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"
    assert len(camera) == 1, f"{len(camera)=}"

    camera_single = camera[0].to(device=vertices.device, convention="opencv")
    intrinsics = camera_single.intrinsics
    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    depth = vertices_camera[:, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = fx * (vertices_camera[:, 0] / depth) + cx
    y = fy * (vertices_camera[:, 1] / depth) + cy

    valid = (
        (depth > 1e-8)
        & (x >= 0.0)
        & (x <= float(image_width - 1))
        & (y >= 0.0)
        & (y <= float(image_height - 1))
    )
    xy = torch.stack([x, y], dim=1)
    return xy, depth, vertices_camera, valid


def _compute_rasterized_visible_vertex_mask(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute rasterized one-view vertex visibility mask.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Bool visibility mask over vertices [V].
    """
    # Input validations
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
    assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"

    device = vertices_camera.device
    vertex_count = vertices_camera.shape[0]
    positive_depth = vertices_camera[:, 2] > 1e-8
    if not torch.any(positive_depth):
        return torch.zeros((vertex_count,), device=device, dtype=torch.bool)

    face_plus1 = _render_camera_face_index_buffer(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    visible_pixels = face_plus1[..., 0] > 0
    visible_vertex_mask = torch.zeros((vertex_count,), device=device, dtype=torch.bool)
    if torch.any(visible_pixels):
        visible_faces = face_plus1[..., 0][visible_pixels].to(dtype=torch.long) - 1
        visible_vertex_indices = faces[visible_faces].reshape(-1)
        visible_vertex_mask[visible_vertex_indices.unique()] = True

    return visible_vertex_mask & positive_depth


def _compute_rasterized_visible_face_mask(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute rasterized one-view face visibility mask.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Bool visibility mask over faces [F].
    """
    # Input validations
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
    assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"

    face_plus1 = _render_camera_face_index_buffer(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    )

    visible_pixels = face_plus1[..., 0] > 0
    visible_face_mask = torch.zeros(
        (faces.shape[0],), device=vertices_camera.device, dtype=torch.bool
    )
    if torch.any(visible_pixels):
        visible_faces = face_plus1[..., 0][visible_pixels].to(dtype=torch.long) - 1
        visible_face_mask[visible_faces.unique()] = True
    return visible_face_mask


def _camera_vertices_to_clip(
    vertices_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert camera-space vertices to clip-space for rasterization.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Clip-space vertices [1, V, 4].
    """
    # Input validations
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
    assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
    assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"

    x_camera = vertices_camera[:, 0]
    y_camera = vertices_camera[:, 1]
    z_camera = vertices_camera[:, 2].clamp(min=1e-6)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x_pixel = fx * (x_camera / z_camera) + cx
    y_pixel = fy * (y_camera / z_camera) + cy
    x_ndc = (x_pixel / float(max(image_width - 1, 1))) * 2.0 - 1.0
    y_ndc = 1.0 - (y_pixel / float(max(image_height - 1, 1))) * 2.0

    z_min = torch.min(z_camera)
    z_max = torch.max(z_camera)
    z_ndc = ((z_camera - z_min) / (z_max - z_min + 1e-6)) * 2.0 - 1.0
    w = torch.ones_like(z_ndc)
    return torch.stack([x_ndc, y_ndc, z_ndc, w], dim=1).unsqueeze(0)


def _vertex_uv_to_clip(
    vertex_uv: torch.Tensor,
    overlap_depth_priority: torch.Tensor,
) -> torch.Tensor:
    """Convert UV coordinates to clip-space positions for UV rasterization.

    Args:
        vertex_uv: Per-vertex UV coordinates [V, 2].
        overlap_depth_priority: Per-vertex scalar priority [V] used to resolve UV overlaps.

    Returns:
        Clip-space UV vertices [1, V, 4].
    """
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(
        overlap_depth_priority, torch.Tensor
    ), f"{type(overlap_depth_priority)=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert overlap_depth_priority.ndim == 1, f"{overlap_depth_priority.shape=}"
    assert (
        overlap_depth_priority.shape[0] == vertex_uv.shape[0]
    ), f"{overlap_depth_priority.shape=} {vertex_uv.shape=}"

    x = vertex_uv[:, 0] * 2.0 - 1.0
    y = 1.0 - vertex_uv[:, 1] * 2.0
    z = (overlap_depth_priority - torch.min(overlap_depth_priority)) / (
        torch.max(overlap_depth_priority) - torch.min(overlap_depth_priority) + 1e-6
    ) * 2.0 - 1.0
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)
