"""Texture extraction utilities for generic triangle meshes."""

from typing import Dict, List, Optional, Tuple, Union

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.validation import validate_rotation_matrix
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
) -> torch.Tensor:
    """Extract texture from multi-view RGB images.

    Returns:
        - Vertex-color path: [V, 3] RGB tensor.
        - UV-map path: [1, H, W, 3] RGB tensor.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert faces is None or isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(images, torch.Tensor) or isinstance(images, list), f"{type(images)=}"
    assert isinstance(cameras, Cameras), f"{type(cameras)=}"
    assert isinstance(representation, str), f"{type(representation)=}"
    assert representation in ("vertex_color", "uv_texture_map"), f"{representation=}"
    assert isinstance(weights, str), f"{type(weights)=}"
    assert weights in ("visible", "normals"), f"{weights=}"
    assert vertex_uv is None or isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert isinstance(default_color, float), f"{type(default_color)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert not isinstance(images, torch.Tensor) or images.ndim == 4
    assert not isinstance(images, list) or len(images) > 0
    assert not isinstance(images, list) or all(isinstance(image, torch.Tensor) for image in images)
    assert faces is not None or weights == "visible"
    assert faces is not None or representation == "vertex_color"
    assert vertex_uv is not None or representation == "vertex_color"
    assert faces is None or (
        faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype in (torch.int32, torch.int64)
    )
    assert vertex_uv is None or (
        vertex_uv.ndim == 2 and vertex_uv.shape[1] == 2 and vertex_uv.shape[0] == vertices.shape[0]
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
        return _extract_vertex_color_from_images(
            vertices=vertices,
            faces=faces,
            images_nchw=images_nchw,
            cameras=cameras,
            weights=weights,
            default_color=default_color,
        )

    assert representation == "uv_texture_map", f"{representation=}"
    assert faces is not None, f"{faces=}"
    assert vertex_uv is not None, f"{vertex_uv=}"
    return _extract_uv_texture_map_from_images(
        vertices=vertices,
        faces=faces,
        images_nchw=images_nchw,
        cameras=cameras,
        weights=weights,
        vertex_uv=vertex_uv,
        texture_size=texture_size,
        default_color=default_color,
    )


def _extract_vertex_color_from_images(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    images_nchw: torch.Tensor,
    cameras: Cameras,
    weights: str,
    default_color: float,
) -> torch.Tensor:
    """Extract fused vertex colors from multiple views.

    Each view contributes projected per-vertex RGB values and per-vertex weights,
    then weighted averaging is applied over views.
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
    weight_denominator = torch.zeros((vertex_count, 1), device=device, dtype=torch.float32)

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
    return vertex_color.clamp(0.0, 1.0).contiguous()


def _extract_vertex_color_from_single_image(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor],
    image: torch.Tensor,
    camera: Cameras,
    weights: str,
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Extract one-view per-vertex RGB and per-vertex weights.

    Returns:
        - texture: [V, 3] RGB colors.
        - weight: [V, 1] weights with visibility already applied.
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
    """Return a binary per-vertex visibility mask for one image/view.

    Output is float32 with values exactly in {0, 1}.
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

    visibility_mask = visibility_bool.to(dtype=torch.float32)
    return visibility_mask


def _compute_v_normals_weights(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Compute per-vertex normal-alignment weights for one view.

    Weight is max(dot(n_cam, v_cam), 0) where both vectors are unit-length.
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

    normals_world = compute_vertex_normals(
        base_vertices=vertices,
        faces=faces,
    ).to(device=vertices.device, dtype=torch.float32)
    normals_world_norm = torch.linalg.norm(normals_world, dim=1)
    normals_world_norm_error = torch.max(torch.abs(normals_world_norm - 1.0))
    assert float(normals_world_norm_error) <= 1.0e-05, f"{float(normals_world_norm_error)=}"

    rotation_w2c = camera[0].w2c[:3, :3]
    validate_rotation_matrix(rotation_w2c)
    normals_camera = normals_world @ rotation_w2c.transpose(0, 1)
    normals_camera_norm = torch.linalg.norm(normals_camera, dim=1)
    normals_camera_norm_error = torch.max(torch.abs(normals_camera_norm - 1.0))
    assert float(normals_camera_norm_error) <= 1.0e-05, f"{float(normals_camera_norm_error)=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    view_direction = F.normalize(-vertices_camera, p=2, dim=1)
    alignment = (normals_camera * view_direction).sum(dim=1).clamp(0.0, 1.0)
    return alignment


def _project_v_colors(
    vertices: torch.Tensor,
    image: torch.Tensor,
    camera: Cameras,
    default_color: float,
) -> torch.Tensor:
    """Project one image onto vertices and return per-vertex RGB colors [V, 3]."""
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
        sampled_color = image[:, y_idx[projection_valid], x_idx[projection_valid]].transpose(0, 1)
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
) -> torch.Tensor:
    """Extract fused UV texture map from multiple views.

    Each view contributes a UV RGB image [1, H, W, 3] and a UV weight map
    [1, H, W, 1], then weighted averaging is applied over views.
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
    assert vertex_uv.shape[0] == vertices.shape[0], f"{vertex_uv.shape=} {vertices.shape=}"

    device = vertices.device
    faces_long = faces.to(device=device, dtype=torch.long).contiguous()
    uv_rasterization_data = _build_uv_rasterization_data(
        vertex_uv=vertex_uv.to(device=device, dtype=torch.float32),
        faces=faces_long,
        texture_size=texture_size,
    )

    uv_numerator = torch.zeros(
        (1, texture_size, texture_size, 3),
        device=device,
        dtype=torch.float32,
    )
    uv_denominator = torch.zeros(
        (1, texture_size, texture_size, 1),
        device=device,
        dtype=torch.float32,
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
    return uv_texture_map.clamp(0.0, 1.0).contiguous()


def _extract_uv_texture_map_from_single_image(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    image: torch.Tensor,
    camera: Cameras,
    weights: str,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Extract one-view UV RGB image and one-view UV weight map.

    Returns:
        - texture: [1, H, W, 3] RGB UV texture from this view.
        - weight: [1, H, W, 1] UV weights from visibility and optional normals.
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

    face_visibility_mask = _compute_f_visibility_mask(
        vertices=vertices,
        faces=faces,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    if weights == "normals":
        face_normals_weight = _compute_f_normals_weights(
            vertices=vertices,
            faces=faces,
            camera=camera,
        )
        face_weight = face_visibility_mask * face_normals_weight
    else:
        face_weight = face_visibility_mask

    uv_texture = _project_f_colors(
        vertices=vertices,
        image=image,
        camera=camera,
        uv_rasterization_data=uv_rasterization_data,
    )
    uv_weight = _rasterize_face_weights_to_uv(
        face_weight=face_weight,
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
) -> torch.Tensor:
    """Return a binary per-face visibility mask for one image/view.

    Output is float32 with values exactly in {0, 1}.
    """
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert isinstance(image_height, int), f"{type(image_height)=}"
    assert isinstance(image_width, int), f"{type(image_width)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert faces.dtype == torch.long, f"{faces.dtype=}"
    assert len(camera) == 1, f"{len(camera)=}"
    assert image_height > 0, f"{image_height=}"
    assert image_width > 0, f"{image_width=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    visible_face_mask = _compute_rasterized_visible_face_mask(
        vertices_camera=vertices_camera,
        faces=faces,
        intrinsics=camera[0].intrinsics,
        image_height=image_height,
        image_width=image_width,
    )

    v0_camera = vertices_camera[faces[:, 0]]
    v1_camera = vertices_camera[faces[:, 1]]
    v2_camera = vertices_camera[faces[:, 2]]
    face_centers_camera = (v0_camera + v1_camera + v2_camera) / 3.0
    face_depth_valid = face_centers_camera[:, 2] > 1e-8
    visibility_mask = (visible_face_mask & face_depth_valid).to(dtype=torch.float32)
    return visibility_mask


def _compute_f_normals_weights(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Compute per-face normal-alignment weights for one view.

    Weight is max(dot(n_cam, v_cam), 0) using face normals and face-center view
    directions in camera coordinates.
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

    v0_world = vertices[faces[:, 0]]
    v1_world = vertices[faces[:, 1]]
    v2_world = vertices[faces[:, 2]]
    face_normals_world = torch.cross(
        v1_world - v0_world,
        v2_world - v0_world,
        dim=1,
    )
    face_normals_world_norm = torch.linalg.norm(
        face_normals_world,
        dim=1,
        keepdim=True,
    )
    assert torch.all(face_normals_world_norm > 0), f"{face_normals_world_norm.min()=}"
    face_normals_world = face_normals_world / face_normals_world_norm

    rotation_w2c = camera[0].w2c[:3, :3]
    validate_rotation_matrix(rotation_w2c)
    face_normals_camera = face_normals_world @ rotation_w2c.transpose(0, 1)
    face_normals_camera_norm = torch.linalg.norm(face_normals_camera, dim=1)
    face_normals_camera_norm_error = torch.max(torch.abs(face_normals_camera_norm - 1.0))
    assert (
        float(face_normals_camera_norm_error) <= 1.0e-05
    ), f"{float(face_normals_camera_norm_error)=}"

    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    v0_camera = vertices_camera[faces[:, 0]]
    v1_camera = vertices_camera[faces[:, 1]]
    v2_camera = vertices_camera[faces[:, 2]]
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
    """Project one image to UV space and return one UV RGB texture [1, H, W, 3]."""
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
    sampled_image_hwc = sampled_image.permute(0, 2, 3, 1).contiguous()
    return sampled_image_hwc


# -----------------------------------------------------------------------------
# Other helpers
# -----------------------------------------------------------------------------


def _build_uv_rasterization_data(
    vertex_uv: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int,
) -> Dict[str, torch.Tensor]:
    """Build reusable UV rasterization tensors (triangles, raster, UV occupancy mask)."""
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"

    tri_i32 = faces.to(device=vertex_uv.device, dtype=torch.int32).contiguous()
    uv_clip = _vertex_uv_to_clip(vertex_uv=vertex_uv).to(
        device=vertex_uv.device,
        dtype=torch.float32,
    )
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
    """Map one-view per-face weights to one-view per-UV-pixel weights."""
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


def _vertices_world_to_camera(
    vertices: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Transform world-space vertices to camera-space vertices for one view."""
    # Input validations
    assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
    assert isinstance(camera, Cameras), f"{type(camera)=}"
    assert vertices.ndim == 2, f"{vertices.shape=}"
    assert vertices.shape[1] == 3, f"{vertices.shape=}"
    assert len(camera) == 1, f"{len(camera)=}"

    camera_single = camera[0].to(device=vertices.device, convention="opencv")
    w2c = camera_single.w2c
    rotation_w2c = w2c[:3, :3]
    translation_w2c = w2c[:3, 3]
    vertices_camera = vertices @ rotation_w2c.transpose(0, 1) + translation_w2c
    return vertices_camera


def _project_vertices_to_image(
    vertices: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space vertices to image pixels.

    Returns xy pixels, depth, camera-space vertices, and in-frame validity mask.
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
    """Rasterize one view and return bool mask of vertices visible in at least one pixel."""
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

    clip_vertices = _camera_vertices_to_clip(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=device, dtype=torch.float32)
    tri_i32 = faces.to(device=device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_vertices.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )

    face_indices = rast_out[0, ..., 3].to(dtype=torch.long) - 1
    visible_pixels = face_indices >= 0
    visible_vertex_mask = torch.zeros((vertex_count,), device=device, dtype=torch.bool)
    if torch.any(visible_pixels):
        visible_faces = face_indices[visible_pixels]
        visible_vertex_indices = faces.to(device=device, dtype=torch.long)[visible_faces].reshape(
            -1
        )
        visible_vertex_mask[visible_vertex_indices.unique()] = True

    return visible_vertex_mask & positive_depth


def _compute_rasterized_visible_face_mask(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Rasterize one view and return bool mask of faces visible in at least one pixel."""
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

    device = vertices_camera.device
    face_count = faces.shape[0]
    clip_vertices = _camera_vertices_to_clip(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=device, dtype=torch.float32)
    tri_i32 = faces.to(device=device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_vertices.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )

    face_indices = rast_out[0, ..., 3].to(dtype=torch.long) - 1
    visible_pixels = face_indices >= 0
    visible_face_mask = torch.zeros((face_count,), device=device, dtype=torch.bool)
    if torch.any(visible_pixels):
        visible_faces = face_indices[visible_pixels].unique()
        visible_face_mask[visible_faces] = True
    return visible_face_mask


def _camera_vertices_to_clip(
    vertices_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert camera-space vertices to clip-space positions for rasterization."""
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
    clip = torch.stack([x_ndc, y_ndc, z_ndc, w], dim=1).unsqueeze(0)
    return clip


def _vertex_uv_to_clip(
    vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Convert UV coordinates to clip-space positions used by UV rasterization."""
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"

    x = vertex_uv[:, 0] * 2.0 - 1.0
    y = 1.0 - vertex_uv[:, 1] * 2.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)
