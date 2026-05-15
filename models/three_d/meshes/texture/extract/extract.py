"""Texture extraction utilities for generic triangle meshes."""

from typing import Any, Dict, List, Optional, Tuple, Union

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.validate import (
    validate_uv_texture_map,
    validate_vertex_color,
)
from models.three_d.meshes.texture.extract.camera_geometry import (
    _project_vertices_to_image,
)
from models.three_d.meshes.texture.extract.normal_weights import (
    _compute_f_normals_weights,
    _compute_v_normals_weights,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility import (
    compute_f_visibility_mask,
)
from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import (
    compute_f_visibility_mask_v2,
)
from models.three_d.meshes.texture.extract.visibility.vertex_visibility import (
    compute_v_visibility_mask,
)
from models.three_d.meshes.texture.extract.weights_cfg import (
    normalize_weights_cfg,
    validate_weights_cfg,
)


def extract_texture_from_images(
    mesh: Union[Mesh, List[Mesh]],
    images: Union[torch.Tensor, List[torch.Tensor]],
    cameras: Cameras,
    weights_cfg: Dict[str, Any] = {},
    texture_size: int = 1024,
    default_color: float = 0.7,
    return_valid_mask: bool = False,
    texel_visibility_method: str = "v1",
    polygon_rast_method: str = "v2",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract texture from multi-view RGB images.

    Args:
        mesh: One shared `Mesh` for all views or one `Mesh` per view.
        images: Multi-view RGB images as [N, 3, H, W] or [N, H, W, 3], or list of [3, H, W].
        cameras: Per-view cameras in OpenCV convention.
        weights_cfg: Per-view fusion weighting configuration dictionary. Supported keys are
            `weights`, `normals_weight_power`, `normals_weight_threshold`,
            `multi_view_robustness`, `robustness_tau`, and
            `first_frame_blending_weight_power`.
        texture_size: UV texture resolution when `mesh` carries UV coordinates.
        default_color: Fallback color for texels/vertices without any valid observation.
        return_valid_mask: Whether to also return a binary valid-observation mask.
        texel_visibility_method: Texel visibility algorithm for UV extraction.
        polygon_rast_method: Step-2 polygon rasterization method for UV extraction.

    Returns:
        If return_valid_mask is False:
            Texture tensor in the representation implied by `mesh`:
            [V, 3] for vertex colors, or [1, texture_size, texture_size, 3] for UV
            texture map in ordinary image row order where row `0` is the image top.
        If return_valid_mask is True:
            Dict containing:
                "texture": texture tensor in selected representation.
                "valid_mask": binary mask of valid observations:
                    [V, 1] for vertex colors, or [1, texture_size, texture_size, 1]
                    for UV texture map in ordinary image row order where row `0` is
                    the image top.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh) or (
            isinstance(mesh, list)
            and mesh != []
            and all(isinstance(item, Mesh) for item in mesh)
        ), (
            "Expected `mesh` to be a `Mesh` or a non-empty list of `Mesh` instances. "
            f"{type(mesh)=}"
        )
        assert isinstance(images, torch.Tensor) or isinstance(images, list), (
            "Expected `images` to be a tensor or a list of tensors. " f"{type(images)=}"
        )
        assert isinstance(cameras, Cameras), (
            "Expected `cameras` to be a `Cameras` instance. " f"{type(cameras)=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        assert isinstance(return_valid_mask, bool), (
            "Expected `return_valid_mask` to be a `bool`. "
            f"{type(return_valid_mask)=}"
        )
        assert isinstance(texel_visibility_method, str), (
            "Expected `texel_visibility_method` to be a `str`. "
            f"{type(texel_visibility_method)=}"
        )
        assert isinstance(polygon_rast_method, str), (
            "Expected `polygon_rast_method` to be a `str`. "
            f"{type(polygon_rast_method)=}"
        )
        assert not isinstance(images, torch.Tensor) or images.ndim == 4, (
            "Expected tensor `images` to have shape `[N, C, H, W]`. " f"{images.shape=}"
        )
        assert not isinstance(images, list) or len(images) > 0, (
            "Expected list `images` to be non-empty. " f"{len(images)=}"
        )
        assert not isinstance(images, list) or all(
            isinstance(image, torch.Tensor) for image in images
        ), (
            "Expected every item in list `images` to be a tensor. "
            f"{[type(image) for image in images]=}"
        )
        assert texel_visibility_method in ("v1", "v2"), (
            "Expected `texel_visibility_method` to be one of the supported texel "
            "visibility methods. "
            f"{texel_visibility_method=}."
        )
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )
        if isinstance(images, list):
            assert len(cameras) == len(images), (
                "Expected one camera per input image. "
                f"{len(cameras)=} {len(images)=}"
            )
        else:
            assert len(cameras) == int(images.shape[0]), (
                "Expected one camera per input image. "
                f"{len(cameras)=} {images.shape[0]=}"
            )
        if isinstance(mesh, list):
            if isinstance(images, list):
                assert len(mesh) == len(images), (
                    "Expected one mesh per view when a mesh list is provided. "
                    f"{len(mesh)=} {len(images)=}"
                )
            else:
                assert len(mesh) == int(images.shape[0]), (
                    "Expected one mesh per view when a mesh list is provided. "
                    f"{len(mesh)=} {images.shape[0]=}"
                )
            for view_mesh in mesh[1:]:
                assert mesh[0].faces.shape == view_mesh.faces.shape, (
                    "Expected all per-view meshes to share one topology. "
                    f"{mesh[0].faces.shape=} {view_mesh.faces.shape=}"
                )
                assert torch.equal(
                    mesh[0].faces.detach().cpu(),
                    view_mesh.faces.detach().cpu(),
                ), (
                    "Expected all per-view meshes to share identical faces. "
                    f"{mesh[0].faces.shape=} {view_mesh.faces.shape=}"
                )
                assert mesh[0].vertices.shape == view_mesh.vertices.shape, (
                    "Expected all per-view meshes to share one vertex layout. "
                    f"{mesh[0].vertices.shape=} {view_mesh.vertices.shape=}"
                )
                assert (mesh[0].vertex_uv is None) == (view_mesh.vertex_uv is None), (
                    "Expected all per-view meshes to agree on whether UV coordinates "
                    "are provided. "
                    f"{mesh[0].vertex_uv is None=} {view_mesh.vertex_uv is None=}"
                )
                if mesh[0].vertex_uv is not None:
                    assert view_mesh.vertex_uv is not None, (
                        "Expected every per-view mesh to provide UV coordinates when "
                        f"the reference mesh does. {view_mesh.vertex_uv=}"
                    )
                    assert mesh[0].vertex_uv.shape == view_mesh.vertex_uv.shape, (
                        "Expected all per-view meshes to share one UV layout. "
                        f"{mesh[0].vertex_uv.shape=} {view_mesh.vertex_uv.shape=}"
                    )
                    assert mesh[0].convention == view_mesh.convention, (
                        "Expected all per-view meshes to share one UV convention. "
                        f"{mesh[0].convention=} {view_mesh.convention=}"
                    )
                    assert torch.equal(
                        mesh[0].vertex_uv.detach().cpu(),
                        view_mesh.vertex_uv.detach().cpu(),
                    ), (
                        "Expected all per-view meshes to share identical UV coordinates. "
                        f"{mesh[0].vertex_uv.shape=} {view_mesh.vertex_uv.shape=}"
                    )
                    assert (mesh[0].face_uvs is None) == (view_mesh.face_uvs is None), (
                        "Expected all per-view meshes to agree on whether `face_uvs` "
                        "is provided. "
                        f"{mesh[0].face_uvs is None=} {view_mesh.face_uvs is None=}"
                    )
                    if mesh[0].face_uvs is not None:
                        assert view_mesh.face_uvs is not None, (
                            "Expected every per-view mesh to provide `face_uvs` when "
                            f"the reference mesh does. {view_mesh.face_uvs=}"
                        )
                        assert mesh[0].face_uvs.shape == view_mesh.face_uvs.shape, (
                            "Expected all per-view meshes to share one UV-face layout. "
                            f"{mesh[0].face_uvs.shape=} {view_mesh.face_uvs.shape=}"
                        )
                        assert torch.equal(
                            mesh[0].face_uvs.detach().cpu(),
                            view_mesh.face_uvs.detach().cpu(),
                        ), (
                            "Expected all per-view meshes to share identical UV-face "
                            "indices. "
                            f"{mesh[0].face_uvs.shape=} {view_mesh.face_uvs.shape=}"
                        )
        validate_weights_cfg(
            weights_cfg=weights_cfg,
        )

    _validate_inputs()

    def _normalize_inputs(
        weights_cfg: Dict[str, Any],
    ) -> Tuple[torch.Tensor, List[Mesh], Dict[str, Any]]:
        """Normalize input arguments.

        Args:
            weights_cfg: Per-view fusion weighting configuration dictionary.

        Returns:
            Normalized local variables for the enclosing function.
        """
        if isinstance(images, list):
            image_stack = torch.stack(images, dim=0)
            image_count = len(images)
        else:
            image_stack = images
            image_count = image_stack.shape[0]
        if isinstance(mesh, Mesh):
            meshes = [mesh for _ in range(image_count)]
        else:
            meshes = list(mesh)
        weights_cfg = normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )
        return (
            image_stack,
            meshes,
            weights_cfg,
        )

    image_stack, meshes, weights_cfg = _normalize_inputs(weights_cfg=weights_cfg)
    representation = meshes[0].texture_mode

    if image_stack.shape[1] == 3:
        images_nchw = image_stack
    else:
        assert image_stack.shape[3] == 3, (
            "Expected channel-last `image_stack` to have three RGB channels. "
            f"{image_stack.shape=}"
        )
        images_nchw = image_stack.permute(0, 3, 1, 2).contiguous()

    if images_nchw.dtype == torch.uint8:
        images_nchw = images_nchw.to(dtype=torch.float32) / 255.0
    else:
        images_nchw = images_nchw.to(dtype=torch.float32)

    device = meshes[0].vertices.device
    meshes = [view_mesh.to(device=device) for view_mesh in meshes]
    images_nchw = images_nchw.to(device=device)
    cameras = cameras.to(device=device, convention="opencv")

    if representation == "vertex_color":
        extracted_vertex_color = _extract_vertex_color_from_images(
            meshes=meshes,
            images_nchw=images_nchw,
            cameras=cameras,
            weights_cfg=weights_cfg,
            default_color=default_color,
        )
        if not return_valid_mask:
            return extracted_vertex_color["texture"]
        return extracted_vertex_color
    if representation == "uv_texture_map":
        meshes = [view_mesh.to(device=device, convention="obj") for view_mesh in meshes]
        extracted_uv_texture_map = _extract_uv_texture_map_from_images(
            meshes=meshes,
            images_nchw=images_nchw,
            cameras=cameras,
            weights_cfg=weights_cfg,
            texture_size=texture_size,
            default_color=default_color,
            texel_visibility_method=texel_visibility_method,
            polygon_rast_method=polygon_rast_method,
        )
        if not return_valid_mask:
            return extracted_uv_texture_map["texture"]
        return extracted_uv_texture_map


def _extract_vertex_color_from_images(
    meshes: List[Mesh],
    images_nchw: torch.Tensor,
    cameras: Cameras,
    weights_cfg: Dict[str, Any],
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Fuse per-view projected vertex colors into one vertex-color tensor.

    Args:
        meshes: Per-view extraction meshes.
        images_nchw: Input RGB images [N, 3, H, W].
        cameras: Per-view cameras.
        weights_cfg: Per-view fusion weighting configuration dictionary.
        default_color: Fallback color for vertices without valid observations.

    Returns:
        Dict with:
            "texture": fused vertex colors [V, 3].
            "valid_mask": binary valid-observation mask [V, 1].
    """

    def _validate_inputs() -> None:
        assert isinstance(meshes, list), (
            "Expected `meshes` to be a list. " f"{type(meshes)=}"
        )
        assert meshes != [], "Expected `meshes` to be non-empty. " f"{meshes=}"
        assert all(isinstance(mesh, Mesh) for mesh in meshes), (
            "Expected every item in `meshes` to be a `Mesh`. "
            f"{[type(mesh) for mesh in meshes]=}"
        )
        assert isinstance(images_nchw, torch.Tensor), (
            "Expected `images_nchw` to be a tensor. " f"{type(images_nchw)=}"
        )
        assert isinstance(cameras, Cameras), (
            "Expected `cameras` to be a `Cameras` instance. " f"{type(cameras)=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        assert images_nchw.ndim == 4, (
            "Expected `images_nchw` to have shape `[N, 3, H, W]`. "
            f"{images_nchw.shape=}"
        )
        assert images_nchw.shape[1] == 3, (
            "Expected `images_nchw` to contain three RGB channels. "
            f"{images_nchw.shape=}"
        )
        assert (
            len(meshes) == images_nchw.shape[0] == len(cameras)
        ), f"{len(meshes)=} {images_nchw.shape=} {len(cameras)=}"
        validate_weights_cfg(
            weights_cfg=weights_cfg,
        )

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize vertex-color extraction inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()

    observations: List[Dict[str, torch.Tensor]] = []
    for view_idx in range(images_nchw.shape[0]):
        observations.append(
            _extract_vertex_color_from_single_image(
                mesh=meshes[view_idx],
                image=images_nchw[view_idx],
                camera=cameras[view_idx : view_idx + 1],
                weights_cfg=weights_cfg,
                default_color=default_color,
            )
        )

    return _fuse_vertex_color_observations(
        observations=observations,
        weights_cfg=weights_cfg,
        default_color=default_color,
    )


def _fuse_vertex_color_observations(
    observations: List[Dict[str, torch.Tensor]],
    weights_cfg: Dict[str, Any],
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Fuse one-view vertex-color observations into one vertex-color tensor.

    Args:
        observations: One-view observation mappings with `texture` and `weight`.
        weights_cfg: Per-view fusion weighting configuration dictionary.
        default_color: Fallback color for vertices without valid observations.

    Returns:
        Dict with fused vertex colors and a valid-observation mask.
    """

    def _validate_inputs() -> None:
        assert isinstance(observations, list), (
            "Expected `observations` to be a list. " f"{type(observations)=}"
        )
        assert observations != [], (
            "Expected `observations` to be non-empty. " f"{observations=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        for observation in observations:
            assert isinstance(observation, dict), (
                "Expected each `observation` to be a dictionary. "
                f"{type(observation)=}"
            )
            assert set(observation.keys()) == {"texture", "weight"}, (
                "Expected one vertex-color observation to contain `texture` and "
                f"`weight`. {observation.keys()=}"
            )
            assert isinstance(observation["texture"], torch.Tensor), (
                "Expected vertex-color observations to store tensor `texture` "
                "values. "
                f"{type(observation['texture'])=}"
            )
            validate_vertex_color(obj=observation["texture"])
            assert isinstance(observation["weight"], torch.Tensor), (
                "Expected vertex-color observations to store tensor `weight` "
                "values. "
                f"{type(observation['weight'])=}"
            )
            assert observation["weight"].ndim == 2, (
                "Expected vertex-color weights to have shape `[V, 1]`. "
                f"{observation['weight'].shape=}"
            )
            assert observation["weight"].shape[1] == 1, (
                "Expected vertex-color weights to have one scalar weight per "
                "vertex. "
                f"{observation['weight'].shape=}"
            )
            assert observation["weight"].dtype == torch.float32, (
                "Expected vertex-color weights to already be float32 before "
                "fusion. "
                f"{observation['weight'].dtype=}"
            )
            assert torch.isfinite(observation["weight"]).all(), (
                "Expected vertex-color weights to contain only finite values. "
                f"{observation['weight'].shape=}"
            )
            assert float(observation["weight"].min().item()) >= 0.0, (
                "Expected vertex-color weights to be non-negative before fusion. "
                f"{float(observation['weight'].min().item())=}"
            )
            assert (
                observation["texture"].shape[0] == observation["weight"].shape[0]
            ), f"{observation['texture'].shape=} {observation['weight'].shape=}"
        validate_weights_cfg(weights_cfg=weights_cfg)

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize vertex-observation fusion inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()
    multi_view_robustness = weights_cfg["multi_view_robustness"]
    robustness_tau = weights_cfg["robustness_tau"]

    device = observations[0]["texture"].device
    vertex_count = observations[0]["texture"].shape[0]
    color_numerator = torch.zeros((vertex_count, 3), device=device, dtype=torch.float32)
    weight_denominator = torch.zeros(
        (vertex_count, 1), device=device, dtype=torch.float32
    )

    if multi_view_robustness == "none":
        for observation in observations:
            texture = observation["texture"].to(device=device)
            weight = observation["weight"].to(device=device)
            color_numerator = color_numerator + texture * weight
            weight_denominator = weight_denominator + weight
    else:
        provisional_numerator = torch.zeros_like(color_numerator)
        provisional_denominator = torch.zeros_like(weight_denominator)
        for observation in observations:
            provisional_texture = observation["texture"].to(device=device)
            provisional_weight = observation["weight"].to(device=device)
            provisional_numerator = (
                provisional_numerator + provisional_texture * provisional_weight
            )
            provisional_denominator = provisional_denominator + provisional_weight

        provisional_vertex_color = torch.full(
            (vertex_count, 3),
            fill_value=default_color,
            device=device,
            dtype=torch.float32,
        )
        validate_vertex_color(obj=provisional_vertex_color)
        provisional_has_weight = provisional_denominator > 0.0
        provisional_vertex_color = torch.where(
            provisional_has_weight.expand_as(provisional_vertex_color),
            provisional_numerator / (provisional_denominator + 1e-6),
            provisional_vertex_color,
        )

        for observation in observations:
            texture = observation["texture"].to(device=device)
            weight = observation["weight"].to(device=device)
            residual = torch.linalg.norm(
                texture - provisional_vertex_color,
                dim=1,
                keepdim=True,
            )
            robust_gate = torch.exp(-torch.square(residual / robustness_tau))
            robust_weight = weight * robust_gate
            color_numerator = color_numerator + texture * robust_weight
            weight_denominator = weight_denominator + robust_weight

    vertex_color = torch.full(
        (vertex_count, 3),
        fill_value=default_color,
        device=device,
        dtype=torch.float32,
    )
    validate_vertex_color(obj=vertex_color)
    has_weight = weight_denominator > 0.0
    vertex_color = torch.where(
        has_weight.expand_as(vertex_color),
        color_numerator / (weight_denominator + 1e-6),
        vertex_color,
    )
    validate_vertex_color(obj=vertex_color)
    return {
        "texture": vertex_color.contiguous(),
        "valid_mask": has_weight.to(dtype=torch.float32).contiguous(),
    }


def _extract_vertex_color_from_single_image(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    weights_cfg: Dict[str, Any],
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Extract one-view vertex colors and corresponding per-vertex weights.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        weights_cfg: One-view weighting configuration dictionary.
        default_color: Fallback color for invalid projections.

    Returns:
        Dict with:
            "texture": Projected vertex RGB colors [V, 3].
            "weight": Per-vertex weights [V, 1].
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(image, torch.Tensor), (
            "Expected `image` to be a tensor. " f"{type(image)=}"
        )
        assert isinstance(camera, Cameras), (
            "Expected `camera` to be a `Cameras` instance. " f"{type(camera)=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        assert image.ndim == 3, (
            "Expected `image` to have shape `[3, H, W]`. " f"{image.shape=}"
        )
        assert image.shape[0] == 3, (
            "Expected `image` to contain three RGB channels. " f"{image.shape=}"
        )
        validate_uv_texture_map(obj=image)
        assert len(camera) == 1, (
            "Expected `camera` to contain exactly one view. " f"{len(camera)=}"
        )
        validate_weights_cfg(
            weights_cfg=weights_cfg,
        )

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize one-view vertex-color extraction inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()
    weights = weights_cfg["weights"]

    visibility_mask = compute_v_visibility_mask(
        mesh=mesh,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    if weights == "normals":
        normals_weight = _compute_v_normals_weights(
            mesh=mesh,
            camera=camera,
            weights_cfg=weights_cfg,
        )
        vertex_weight = visibility_mask * normals_weight
    else:
        vertex_weight = visibility_mask

    vertex_color = _project_v_colors(
        mesh=mesh,
        image=image,
        camera=camera,
        default_color=default_color,
    )
    return {
        "texture": vertex_color,
        "weight": vertex_weight.unsqueeze(1),
    }


def _project_v_colors(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    default_color: float,
) -> torch.Tensor:
    """Project one image to vertices and sample per-vertex RGB colors.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        default_color: Fallback color for invalid projections.

    Returns:
        Vertex RGB colors with shape [V, 3].
    """

    xy, _depth, _vertices_camera, projection_valid = _project_vertices_to_image(
        vertices=mesh.vertices,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )

    vertex_count = mesh.vertices.shape[0]
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
    meshes: List[Mesh],
    images_nchw: torch.Tensor,
    cameras: Cameras,
    weights_cfg: Dict[str, Any],
    texture_size: int,
    default_color: float,
    texel_visibility_method: str,
    polygon_rast_method: str = "v2",
) -> Dict[str, torch.Tensor]:
    """Fuse per-view UV observations into one UV texture map.

    Args:
        meshes: Per-view extraction meshes.
        images_nchw: Input RGB images [N, 3, H, W].
        cameras: Per-view cameras.
        weights_cfg: Per-view fusion weighting configuration dictionary.
        texture_size: UV texture resolution.
        default_color: Fallback color for UV pixels without valid observations.
        texel_visibility_method: Texel visibility algorithm, `"v1"` or `"v2"`.
        polygon_rast_method: Step-2 polygon rasterization method.

    Returns:
        Dict with:
            "texture": fused UV texture map [1, T, T, 3].
            "valid_mask": binary valid-observation mask [1, T, T, 1].
    """

    def _validate_inputs() -> None:
        assert isinstance(meshes, list), (
            "Expected `meshes` to be a list. " f"{type(meshes)=}"
        )
        assert meshes != [], "Expected `meshes` to be non-empty. " f"{meshes=}"
        assert all(isinstance(mesh, Mesh) for mesh in meshes), (
            "Expected every item in `meshes` to be a `Mesh`. "
            f"{[type(mesh) for mesh in meshes]=}"
        )
        assert isinstance(images_nchw, torch.Tensor), (
            "Expected `images_nchw` to be a tensor. " f"{type(images_nchw)=}"
        )
        assert isinstance(cameras, Cameras), (
            "Expected `cameras` to be a `Cameras` instance. " f"{type(cameras)=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        assert isinstance(texel_visibility_method, str), (
            "Expected `texel_visibility_method` to be a `str`. "
            f"{type(texel_visibility_method)=}"
        )
        assert isinstance(polygon_rast_method, str), (
            "Expected `polygon_rast_method` to be a `str`. "
            f"{type(polygon_rast_method)=}"
        )
        assert images_nchw.ndim == 4, (
            "Expected `images_nchw` to have shape `[N, 3, H, W]`. "
            f"{images_nchw.shape=}"
        )
        assert images_nchw.shape[1] == 3, (
            "Expected `images_nchw` to contain three RGB channels. "
            f"{images_nchw.shape=}"
        )
        assert (
            len(meshes) == len(cameras) == images_nchw.shape[0]
        ), f"{len(meshes)=} {len(cameras)=} {images_nchw.shape=}"
        assert meshes[0].vertex_uv is not None, (
            "Expected the reference mesh to provide UV coordinates for UV extraction. "
            f"{meshes[0].vertex_uv=}"
        )
        assert texel_visibility_method in ("v1", "v2"), (
            "Expected `texel_visibility_method` to be one of the supported texel "
            "visibility methods. "
            f"{texel_visibility_method=}."
        )
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )
        validate_weights_cfg(
            weights_cfg=weights_cfg,
        )

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize UV-texture extraction inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()

    reference_mesh = meshes[0]
    uv_rasterization_data = _build_uv_rasterization_data(
        mesh=reference_mesh,
        texture_size=texture_size,
    )
    observations: List[Dict[str, torch.Tensor]] = []
    for view_idx in range(images_nchw.shape[0]):
        observations.append(
            _extract_uv_texture_map_from_single_image(
                mesh=meshes[view_idx],
                image=images_nchw[view_idx],
                camera=cameras[view_idx : view_idx + 1],
                weights_cfg=weights_cfg,
                uv_rasterization_data=uv_rasterization_data,
                texel_visibility_method=texel_visibility_method,
                polygon_rast_method=polygon_rast_method,
            )
        )

    return _fuse_uv_texture_observations(
        observations=observations,
        weights_cfg=weights_cfg,
        default_color=default_color,
    )


def _fuse_uv_texture_observations(
    observations: List[Dict[str, torch.Tensor]],
    weights_cfg: Dict[str, Any],
    default_color: float,
) -> Dict[str, torch.Tensor]:
    """Fuse one-view UV observations into one UV texture map.

    Args:
        observations: One-view observation mappings with `texture` and `weight`.
        weights_cfg: Per-view fusion weighting configuration dictionary.
        default_color: Fallback color for UV texels without valid observations.

    Returns:
        Dict with fused UV texture and a valid-observation mask in ordinary image
        row order where row `0` is the image top.
    """

    def _validate_inputs() -> None:
        assert isinstance(observations, list), (
            "Expected `observations` to be a list. " f"{type(observations)=}"
        )
        assert observations != [], (
            "Expected `observations` to be non-empty. " f"{observations=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(default_color, float), (
            "Expected `default_color` to be a `float`. " f"{type(default_color)=}"
        )
        for observation in observations:
            assert isinstance(observation, dict), (
                "Expected each `observation` to be a dictionary. "
                f"{type(observation)=}"
            )
            assert set(observation.keys()) == {"texture", "weight"}, (
                "Expected one UV observation to contain `texture` and `weight`. "
                f"{observation.keys()=}"
            )
            assert isinstance(observation["texture"], torch.Tensor), (
                "Expected UV observations to store tensor `texture` values. "
                f"{type(observation['texture'])=}"
            )
            validate_uv_texture_map(obj=observation["texture"])
            assert isinstance(observation["weight"], torch.Tensor), (
                "Expected UV observations to store tensor `weight` values. "
                f"{type(observation['weight'])=}"
            )
            assert observation["weight"].ndim == 4, (
                "Expected UV weights to have shape `[1, H, W, 1]`. "
                f"{observation['weight'].shape=}"
            )
            assert observation["weight"].shape[0] == 1, (
                "Expected UV weights to keep batch size 1. "
                f"{observation['weight'].shape=}"
            )
            assert observation["weight"].shape[3] == 1, (
                "Expected UV weights to store one scalar weight per texel. "
                f"{observation['weight'].shape=}"
            )
            assert observation["weight"].dtype == torch.float32, (
                "Expected UV weights to already be float32 before fusion. "
                f"{observation['weight'].dtype=}"
            )
            assert torch.isfinite(observation["weight"]).all(), (
                "Expected UV weights to contain only finite values. "
                f"{observation['weight'].shape=}"
            )
            assert float(observation["weight"].min().item()) >= 0.0, (
                "Expected UV weights to be non-negative before fusion. "
                f"{float(observation['weight'].min().item())=}"
            )
            assert (
                observation["texture"].shape[1] == observation["weight"].shape[1]
            ), f"{observation['texture'].shape=} {observation['weight'].shape=}"
            assert (
                observation["texture"].shape[2] == observation["weight"].shape[2]
            ), f"{observation['texture'].shape=} {observation['weight'].shape=}"
        validate_weights_cfg(weights_cfg=weights_cfg)

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize UV-observation fusion inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()
    multi_view_robustness = weights_cfg["multi_view_robustness"]
    robustness_tau = weights_cfg["robustness_tau"]

    device = observations[0]["texture"].device
    texture_height = observations[0]["texture"].shape[1]
    texture_width = observations[0]["texture"].shape[2]
    uv_numerator = torch.zeros(
        (1, texture_height, texture_width, 3),
        device=device,
        dtype=torch.float32,
    )
    uv_denominator = torch.zeros(
        (1, texture_height, texture_width, 1),
        device=device,
        dtype=torch.float32,
    )

    if multi_view_robustness == "none":
        for observation in observations:
            texture = observation["texture"].to(device=device)
            weight = observation["weight"].to(device=device)
            uv_numerator = uv_numerator + texture * weight
            uv_denominator = uv_denominator + weight
    else:
        provisional_uv_numerator = torch.zeros_like(uv_numerator)
        provisional_uv_denominator = torch.zeros_like(uv_denominator)
        for observation in observations:
            provisional_uv_texture = observation["texture"].to(device=device)
            provisional_uv_weight = observation["weight"].to(device=device)
            provisional_uv_numerator = (
                provisional_uv_numerator
                + provisional_uv_texture * provisional_uv_weight
            )
            provisional_uv_denominator = (
                provisional_uv_denominator + provisional_uv_weight
            )

        provisional_uv_texture_map = torch.full(
            (1, texture_height, texture_width, 3),
            fill_value=default_color,
            device=device,
            dtype=torch.float32,
        )
        validate_uv_texture_map(obj=provisional_uv_texture_map)
        provisional_uv_has_weight = provisional_uv_denominator > 0.0
        provisional_uv_texture_map = torch.where(
            provisional_uv_has_weight.expand_as(provisional_uv_texture_map),
            provisional_uv_numerator / (provisional_uv_denominator + 1e-6),
            provisional_uv_texture_map,
        )

        for observation in observations:
            texture = observation["texture"].to(device=device)
            weight = observation["weight"].to(device=device)
            residual = torch.linalg.norm(
                texture - provisional_uv_texture_map,
                dim=3,
                keepdim=True,
            )
            robust_gate = torch.exp(-torch.square(residual / robustness_tau))
            robust_weight = weight * robust_gate
            uv_numerator = uv_numerator + texture * robust_weight
            uv_denominator = uv_denominator + robust_weight

    uv_texture_map = torch.full(
        (1, texture_height, texture_width, 3),
        fill_value=default_color,
        device=device,
        dtype=torch.float32,
    )
    validate_uv_texture_map(obj=uv_texture_map)
    has_weight = uv_denominator > 0.0
    uv_texture_map = torch.where(
        has_weight.expand_as(uv_texture_map),
        uv_numerator / (uv_denominator + 1e-6),
        uv_texture_map,
    )
    validate_uv_texture_map(obj=uv_texture_map)
    return {
        "texture": uv_texture_map.contiguous(),
        "valid_mask": has_weight.to(dtype=torch.float32).contiguous(),
    }


def _extract_uv_texture_map_from_single_image(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    weights_cfg: Dict[str, Any],
    uv_rasterization_data: Dict[str, torch.Tensor],
    texel_visibility_method: str = "v1",
    polygon_rast_method: str = "v2",
) -> Dict[str, torch.Tensor]:
    """Extract one-view UV texture observation and UV weight map.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        weights_cfg: One-view weighting configuration dictionary.
        uv_rasterization_data: Precomputed UV rasterization tensors.
        texel_visibility_method: Texel visibility algorithm, `"v1"` or `"v2"`.
        polygon_rast_method: Step-2 polygon rasterization method.

    Returns:
        Dict with normalized `texture` and `weight` in ordinary image row order
        where row `0` is the image top.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(image, torch.Tensor), (
            "Expected `image` to be a tensor. " f"{type(image)=}"
        )
        assert isinstance(camera, Cameras), (
            "Expected `camera` to be a `Cameras` instance. " f"{type(camera)=}"
        )
        assert isinstance(weights_cfg, dict), (
            "Expected `weights_cfg` to be a dictionary. " f"{type(weights_cfg)=}"
        )
        assert isinstance(uv_rasterization_data, dict), (
            "Expected `uv_rasterization_data` to be a dictionary. "
            f"{type(uv_rasterization_data)=}"
        )
        assert isinstance(texel_visibility_method, str), (
            "Expected `texel_visibility_method` to be a `str`. "
            f"{type(texel_visibility_method)=}"
        )
        assert isinstance(polygon_rast_method, str), (
            "Expected `polygon_rast_method` to be a `str`. "
            f"{type(polygon_rast_method)=}"
        )
        assert image.ndim == 3, (
            "Expected `image` to have shape `[3, H, W]`. " f"{image.shape=}"
        )
        assert image.shape[0] == 3, (
            "Expected `image` to contain three RGB channels. " f"{image.shape=}"
        )
        validate_uv_texture_map(obj=image)
        assert len(camera) == 1, (
            "Expected `camera` to contain exactly one view. " f"{len(camera)=}"
        )
        assert texel_visibility_method in ("v1", "v2"), (
            "Expected `texel_visibility_method` to be one of the supported texel "
            "visibility methods. "
            f"{texel_visibility_method=}."
        )
        assert polygon_rast_method in ("v1", "v2"), (
            "Expected `polygon_rast_method` to be one of the supported polygon "
            "rasterization methods. "
            f"{polygon_rast_method=}."
        )
        validate_weights_cfg(
            weights_cfg=weights_cfg,
        )

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize one-view UV-texture extraction inputs.

        Args:
            None.

        Returns:
            Normalized weights config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="visible",
        )

    weights_cfg = _normalize_inputs()
    weights = weights_cfg["weights"]

    if texel_visibility_method == "v1":
        uv_visibility_mask = compute_f_visibility_mask(
            vertices=mesh.vertices,
            faces=mesh.faces,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            uv_rasterization_data=uv_rasterization_data,
            polygon_rast_method=polygon_rast_method,
        )
    else:
        assert texel_visibility_method == "v2", (
            "Expected `texel_visibility_method` to be `v1` or `v2`. "
            f"{texel_visibility_method=}"
        )
        uv_visibility_mask = compute_f_visibility_mask_v2(
            vertices=mesh.vertices,
            faces=mesh.faces,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            uv_rasterization_data=uv_rasterization_data,
        )
    if weights == "normals":
        face_normals_weight = _compute_f_normals_weights(
            mesh=mesh,
            camera=camera,
            weights_cfg=weights_cfg,
        )
        uv_normals_weight = _rasterize_face_weights_to_uv(
            face_weight=face_normals_weight,
            uv_rasterization_data=uv_rasterization_data,
        )
        uv_weight = uv_visibility_mask * uv_normals_weight
    else:
        uv_weight = uv_visibility_mask

    uv_texture = _project_f_colors(
        mesh=mesh,
        image=image,
        camera=camera,
        uv_rasterization_data=uv_rasterization_data,
    )
    return {
        "texture": torch.flip(uv_texture, dims=[1]).contiguous(),
        "weight": torch.flip(uv_weight.to(dtype=torch.float32), dims=[1]).contiguous(),
    }


def _project_f_colors(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    uv_rasterization_data: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Project one image into UV space using rasterized UV correspondence.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        uv_rasterization_data: Precomputed UV rasterization tensors.

    Returns:
        One-view UV RGB image with shape [1, T, T, 3].
    """

    def _validate_inputs() -> None:
        assert isinstance(uv_rasterization_data, dict), (
            "Expected `uv_rasterization_data` to be a dictionary. "
            f"{type(uv_rasterization_data)=}"
        )
        assert "tri_i32" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `tri_i32`. "
            f"{uv_rasterization_data.keys()=}"
        )
        assert "rast_out" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `rast_out`. "
            f"{uv_rasterization_data.keys()=}"
        )
        assert "raster_vertex_indices" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `raster_vertex_indices`. "
            f"{uv_rasterization_data.keys()=}"
        )

    _validate_inputs()

    def _interpolate_uv_texel_image_coords(
        projected_vertex_xy: torch.Tensor,
        uv_rasterization_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Interpolate image-space coordinates for every occupied UV texel.

        Args:
            projected_vertex_xy: Image-space mesh-vertex coordinates `[V, 2]`.
            uv_rasterization_data: Precomputed UV rasterization tensors.

        Returns:
            Interpolated image-space UV texel coordinates `[1, T, T, 2]`.
        """

        def _validate_inputs() -> None:
            assert isinstance(projected_vertex_xy, torch.Tensor), (
                "Expected `projected_vertex_xy` to be a tensor. "
                f"{type(projected_vertex_xy)=}"
            )
            assert projected_vertex_xy.ndim == 2, (
                "Expected `projected_vertex_xy` to have shape `[V, 2]`. "
                f"{projected_vertex_xy.shape=}"
            )
            assert projected_vertex_xy.shape[1] == 2, (
                "Expected `projected_vertex_xy` to have shape `[V, 2]`. "
                f"{projected_vertex_xy.shape=}"
            )

        _validate_inputs()

        tri_i32 = uv_rasterization_data["tri_i32"]
        rast_out = uv_rasterization_data["rast_out"]
        raster_vertex_indices = uv_rasterization_data["raster_vertex_indices"]
        assert isinstance(raster_vertex_indices, torch.Tensor), (
            "Expected `raster_vertex_indices` to be a tensor. "
            f"{type(raster_vertex_indices)=}"
        )
        raster_xy = projected_vertex_xy[raster_vertex_indices]
        interpolated_uv_xy, _ = dr.interpolate(
            attr=raster_xy.unsqueeze(0).contiguous(),
            rast=rast_out,
            tri=tri_i32,
        )
        return interpolated_uv_xy

    def _sample_uv_texel_colors_from_source_image(
        interpolated_uv_xy: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Sample source-image colors at interpolated UV texel image coordinates.

        Args:
            interpolated_uv_xy: Image-space UV texel coordinates `[1, T, T, 2]`.
            image: One RGB image `[3, H, W]`.

        Returns:
            Sampled UV texture map `[1, T, T, 3]`.
        """

        def _validate_inputs() -> None:
            assert isinstance(interpolated_uv_xy, torch.Tensor), (
                "Expected `interpolated_uv_xy` to be a tensor. "
                f"{type(interpolated_uv_xy)=}"
            )
            assert interpolated_uv_xy.ndim == 4, (
                "Expected `interpolated_uv_xy` to have shape `[1, T, T, 2]`. "
                f"{interpolated_uv_xy.shape=}"
            )
            assert interpolated_uv_xy.shape[0] == 1, (
                "Expected `interpolated_uv_xy` to keep batch size 1. "
                f"{interpolated_uv_xy.shape=}"
            )
            assert interpolated_uv_xy.shape[3] == 2, (
                "Expected `interpolated_uv_xy` to store 2D image coordinates. "
                f"{interpolated_uv_xy.shape=}"
            )

        _validate_inputs()

        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        grid_x = interpolated_uv_xy[..., 0] / float(max(image_width - 1, 1)) * 2.0 - 1.0
        grid_y = (
            interpolated_uv_xy[..., 1] / float(max(image_height - 1, 1)) * 2.0 - 1.0
        )
        sampling_grid = torch.stack([grid_x, grid_y], dim=-1).contiguous()
        sampled_image = F.grid_sample(
            input=image.unsqueeze(0),
            grid=sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled_image.permute(0, 2, 3, 1).contiguous()

    xy, _depth, _vertices_camera, _valid = _project_vertices_to_image(
        vertices=mesh.vertices,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    interpolated_uv_xy = _interpolate_uv_texel_image_coords(
        projected_vertex_xy=xy,
        uv_rasterization_data=uv_rasterization_data,
    )
    uv_texture = _sample_uv_texel_colors_from_source_image(
        interpolated_uv_xy=interpolated_uv_xy,
        image=image,
    )
    validate_uv_texture_map(obj=uv_texture)
    return uv_texture.contiguous()


# -----------------------------------------------------------------------------
# Other helpers
# -----------------------------------------------------------------------------


def _build_uv_rasterization_data(
    mesh: Mesh,
    texture_size: int,
) -> Dict[str, torch.Tensor]:
    """Build reusable UV rasterization tensors for UV-space operations.

    Args:
        mesh: Mesh with vertex_uv and faces (and optionally face_uvs).
        texture_size: UV texture resolution.

    Returns:
        Dict containing:
            "tri_i32": Triangle index tensor with shape [F, 3] and dtype
                `torch.int32`, where each row stores the three triangle-soup
                vertex indices of one UV-rasterized triangle.
            "rast_out": UV rasterization output [1, T, T, 4].
            "uv_mask": UV occupancy mask [1, T, T, 1].
            "raster_vertex_indices": Original mesh-vertex index for each
                triangle-soup vertex [Vr].
            "raster_face_indices": Original mesh-face index for each
                UV-rasterized triangle [Fr].
            "camera_attr_vertex_uv": Per-face local UV attributes [F * 3, 2]
                for camera-space interpolation.
            "camera_attr_tri_i32": Triangle index tensor [F, 3] with dtype
                `torch.int32` for camera-space UV interpolation.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )
        assert mesh.vertex_uv is not None, (
            "Expected mesh to have vertex_uv for UV rasterization. "
            f"{mesh.vertex_uv is None=}"
        )

    _validate_inputs()

    vertex_uv = mesh.vertex_uv
    faces = mesh.faces
    face_uvs = mesh.face_uvs

    uv_rasterization_mesh = _build_uv_rasterization_mesh(
        vertex_uv=vertex_uv,
        faces=faces,
        face_uvs=face_uvs,
    )
    camera_uv_interpolation_data = _build_camera_uv_interpolation_data(
        vertex_uv=vertex_uv,
        faces=faces,
        face_uvs=face_uvs,
    )
    tri_i32 = uv_rasterization_mesh["tri_i32"]
    raster_vertex_uv = uv_rasterization_mesh["raster_vertex_uv"]
    uv_clip = _vertex_uv_to_clip(
        vertex_uv=raster_vertex_uv,
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
        "raster_vertex_indices": uv_rasterization_mesh["raster_vertex_indices"],
        "raster_face_indices": uv_rasterization_mesh["raster_face_indices"],
        "camera_attr_vertex_uv": camera_uv_interpolation_data["camera_attr_vertex_uv"],
        "camera_attr_tri_i32": camera_uv_interpolation_data["camera_attr_tri_i32"],
    }


def _build_uv_rasterization_mesh(
    vertex_uv: torch.Tensor,
    faces: torch.Tensor,
    face_uvs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build a seam-safe UV triangle soup for UV rasterization.

    Args:
        vertex_uv: Per-vertex UV coordinates [V, 2].
        faces: Mesh faces [F, 3].
        face_uvs: Optional face-to-UV indices [F, 3]. When omitted, reuse `faces`.

    Returns:
        Dict containing:
            "raster_vertex_uv": Triangle-soup UV coordinates [Vr, 2].
            "tri_i32": Triangle-soup indices [Fr, 3] with dtype `torch.int32`.
            "raster_vertex_indices": Original mesh-vertex indices [Vr].
            "raster_face_indices": Original mesh-face index for each UV triangle [Fr].
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_uv, torch.Tensor), (
            "Expected `vertex_uv` to be a tensor. " f"{type(vertex_uv)=}"
        )
        assert isinstance(faces, torch.Tensor), (
            "Expected `faces` to be a tensor. " f"{type(faces)=}"
        )
        assert face_uvs is None or isinstance(face_uvs, torch.Tensor), (
            "Expected `face_uvs` to be `None` or a tensor. " f"{type(face_uvs)=}"
        )
        assert vertex_uv.ndim == 2, (
            "Expected `vertex_uv` to have shape `[V, 2]`. " f"{vertex_uv.shape=}"
        )
        assert vertex_uv.shape[1] == 2, (
            "Expected `vertex_uv` to have shape `[V, 2]`. " f"{vertex_uv.shape=}"
        )
        assert faces.ndim == 2, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}"
        )
        assert faces.shape[1] == 3, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}"
        )
        assert face_uvs is None or face_uvs.shape == faces.shape, (
            "Expected `face_uvs` to align with `faces` when provided. "
            f"{face_uvs.shape=} {faces.shape=}"
        )

    _validate_inputs()

    faces_long = faces.to(device=vertex_uv.device, dtype=torch.long).contiguous()
    if face_uvs is None:
        face_uvs_long = faces_long
    else:
        face_uvs_long = face_uvs.to(
            device=vertex_uv.device, dtype=torch.long
        ).contiguous()
    face_vertex_uv = vertex_uv[face_uvs_long]
    face_u = face_vertex_uv[:, :, 0]
    seam_face_mask = (face_u.max(dim=1).values - face_u.min(dim=1).values) > 0.5

    raster_vertex_uv_chunks: List[torch.Tensor] = []
    raster_vertex_index_chunks: List[torch.Tensor] = []
    raster_face_index_chunks: List[torch.Tensor] = []

    def _append_triangles(
        face_indices: torch.Tensor,
        face_uv: torch.Tensor,
    ) -> None:
        """Append one batch of UV triangles to the triangle-soup buffers.

        Args:
            face_indices: Original face indices [K].
            face_uv: UV coordinates for those faces [K, 3, 2].

        Returns:
            None.
        """

        def _validate_inputs() -> None:
            assert isinstance(face_indices, torch.Tensor), (
                "Expected `face_indices` to be a tensor. " f"{type(face_indices)=}"
            )
            assert isinstance(face_uv, torch.Tensor), (
                "Expected `face_uv` to be a tensor. " f"{type(face_uv)=}"
            )
            assert face_indices.ndim == 1, (
                "Expected `face_indices` to have shape `[K]`. " f"{face_indices.shape=}"
            )
            assert face_uv.ndim == 3, (
                "Expected `face_uv` to have shape `[K, 3, 2]`. " f"{face_uv.shape=}"
            )
            assert face_uv.shape[0] == face_indices.shape[0], (
                "Expected `face_uv` to align with `face_indices`. "
                f"{face_uv.shape=} {face_indices.shape=}"
            )
            assert face_uv.shape[1] == 3, (
                "Expected `face_uv` to provide three UV vertices per face. "
                f"{face_uv.shape=}"
            )
            assert face_uv.shape[2] == 2, (
                "Expected `face_uv` to store 2D UV coordinates. " f"{face_uv.shape=}"
            )

        _validate_inputs()

        if face_indices.numel() == 0:
            return

        raster_vertex_uv_chunks.append(
            face_uv.reshape(-1, 2).to(device=vertex_uv.device, dtype=torch.float32)
        )
        raster_vertex_index_chunks.append(faces_long[face_indices].reshape(-1))
        raster_face_index_chunks.append(face_indices)

    non_seam_face_indices = torch.nonzero(~seam_face_mask, as_tuple=False).reshape(-1)
    _append_triangles(
        face_indices=non_seam_face_indices,
        face_uv=face_vertex_uv[non_seam_face_indices],
    )

    seam_face_indices = torch.nonzero(seam_face_mask, as_tuple=False).reshape(-1)
    if seam_face_indices.numel() > 0:
        seam_face_uv = face_vertex_uv[seam_face_indices].clone()
        seam_face_u = seam_face_uv[:, :, 0]
        seam_face_uv[:, :, 0] = torch.where(
            seam_face_u < 0.5,
            seam_face_u + 1.0,
            seam_face_u,
        )
        _append_triangles(
            face_indices=seam_face_indices,
            face_uv=seam_face_uv,
        )
        seam_face_uv_wrapped = seam_face_uv.clone()
        seam_face_uv_wrapped[:, :, 0] = seam_face_uv_wrapped[:, :, 0] - 1.0
        _append_triangles(
            face_indices=seam_face_indices,
            face_uv=seam_face_uv_wrapped,
        )

    assert (
        len(raster_vertex_uv_chunks) > 0
    ), "Failed to build UV rasterization mesh: no raster triangles were generated."
    raster_vertex_uv = torch.cat(raster_vertex_uv_chunks, dim=0).contiguous()
    raster_vertex_indices = torch.cat(raster_vertex_index_chunks, dim=0).contiguous()
    raster_face_indices = torch.cat(raster_face_index_chunks, dim=0).contiguous()
    tri_i32 = torch.arange(
        start=0,
        end=raster_vertex_uv.shape[0],
        device=vertex_uv.device,
        dtype=torch.int32,
    ).reshape(-1, 3)
    assert (
        tri_i32.shape[0] == raster_face_indices.shape[0]
    ), f"{tri_i32.shape=} {raster_face_indices.shape=}"

    return {
        "raster_vertex_uv": raster_vertex_uv,
        "tri_i32": tri_i32,
        "raster_vertex_indices": raster_vertex_indices,
        "raster_face_indices": raster_face_indices,
    }


def _build_camera_uv_interpolation_data(
    vertex_uv: torch.Tensor,
    faces: torch.Tensor,
    face_uvs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build seam-safe per-face UV attributes for camera-space interpolation.

    Args:
        vertex_uv: Per-vertex UV coordinates [V, 2].
        faces: Mesh faces [F, 3].
        face_uvs: Optional face-to-UV indices [F, 3]. When omitted, reuse `faces`.

    Returns:
        Dict containing:
            "camera_attr_vertex_uv": Per-face local UV attributes [F * 3, 2].
            "camera_attr_tri_i32": Triangle index tensor [F, 3] with dtype
                `torch.int32`, where row `f` corresponds to original face `f`.
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_uv, torch.Tensor), (
            "Expected `vertex_uv` to be a tensor. " f"Got {type(vertex_uv)=}."
        )
        assert isinstance(faces, torch.Tensor), (
            "Expected `faces` to be a tensor. " f"Got {type(faces)=}."
        )
        assert face_uvs is None or isinstance(face_uvs, torch.Tensor), (
            "Expected `face_uvs` to be `None` or a tensor. " f"Got {type(face_uvs)=}."
        )
        assert vertex_uv.ndim == 2, (
            "Expected `vertex_uv` to have shape [V, 2]. " f"Got {vertex_uv.shape=}."
        )
        assert vertex_uv.shape[1] == 2, (
            "Expected `vertex_uv` to have shape [V, 2]. " f"Got {vertex_uv.shape=}."
        )
        assert faces.ndim == 2, (
            "Expected `faces` to have shape [F, 3]. " f"Got {faces.shape=}."
        )
        assert faces.shape[1] == 3, (
            "Expected `faces` to have shape [F, 3]. " f"Got {faces.shape=}."
        )
        assert face_uvs is None or face_uvs.shape == faces.shape, (
            "Expected `face_uvs` to align with `faces` when provided. "
            f"{face_uvs.shape=} {faces.shape=}"
        )

    _validate_inputs()

    faces_long = faces.to(device=vertex_uv.device, dtype=torch.long).contiguous()
    if face_uvs is None:
        face_uvs_long = faces_long
    else:
        face_uvs_long = face_uvs.to(
            device=vertex_uv.device, dtype=torch.long
        ).contiguous()
    face_vertex_uv = vertex_uv[face_uvs_long].clone()
    face_u = face_vertex_uv[:, :, 0]
    seam_face_mask = (face_u.max(dim=1).values - face_u.min(dim=1).values) > 0.5
    if torch.any(seam_face_mask):
        seam_face_u = face_vertex_uv[seam_face_mask, :, 0]
        face_vertex_uv[seam_face_mask, :, 0] = torch.where(
            seam_face_u < 0.5,
            seam_face_u + 1.0,
            seam_face_u,
        )
    camera_attr_vertex_uv = face_vertex_uv.reshape(-1, 2).contiguous()
    camera_attr_tri_i32 = torch.arange(
        start=0,
        end=camera_attr_vertex_uv.shape[0],
        device=vertex_uv.device,
        dtype=torch.int32,
    ).reshape(-1, 3)
    assert camera_attr_tri_i32.shape[0] == faces.shape[0], (
        "Expected one camera UV triangle per original face. "
        f"Got {camera_attr_tri_i32.shape=} {faces.shape=}."
    )
    return {
        "camera_attr_vertex_uv": camera_attr_vertex_uv,
        "camera_attr_tri_i32": camera_attr_tri_i32,
    }


def _vertex_uv_to_clip(
    vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Convert UV coordinates to clip-space positions for UV rasterization.

    Args:
        vertex_uv: Per-vertex UV coordinates [V, 2].

    Returns:
        Clip-space UV vertices [1, V, 4].
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_uv, torch.Tensor), (
            "Expected `vertex_uv` to be a tensor. " f"{type(vertex_uv)=}"
        )
        assert vertex_uv.ndim == 2, (
            "Expected `vertex_uv` to have shape `[V, 2]`. " f"{vertex_uv.shape=}"
        )
        assert vertex_uv.shape[1] == 2, (
            "Expected `vertex_uv` to have shape `[V, 2]`. " f"{vertex_uv.shape=}"
        )

    _validate_inputs()

    x = vertex_uv[:, 0] * 2.0 - 1.0
    y = vertex_uv[:, 1] * 2.0 - 1.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)


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

    def _validate_inputs() -> None:
        assert isinstance(face_weight, torch.Tensor), (
            "Expected `face_weight` to be a tensor. " f"{type(face_weight)=}"
        )
        assert isinstance(uv_rasterization_data, dict), (
            "Expected `uv_rasterization_data` to be a dictionary. "
            f"{type(uv_rasterization_data)=}"
        )
        assert "rast_out" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `rast_out`. "
            f"{uv_rasterization_data.keys()=}"
        )
        assert "uv_mask" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `uv_mask`. "
            f"{uv_rasterization_data.keys()=}"
        )
        assert "raster_face_indices" in uv_rasterization_data, (
            "Expected `uv_rasterization_data` to contain `raster_face_indices`. "
            f"{uv_rasterization_data.keys()=}"
        )
        assert face_weight.ndim == 1, (
            "Expected `face_weight` to have shape `[F]`. " f"{face_weight.shape=}"
        )

    _validate_inputs()

    rast_out = uv_rasterization_data["rast_out"]
    uv_mask = uv_rasterization_data["uv_mask"]
    raster_face_indices = uv_rasterization_data["raster_face_indices"]
    assert isinstance(rast_out, torch.Tensor), (
        "Expected `rast_out` to be a tensor. " f"{type(rast_out)=}"
    )
    assert isinstance(uv_mask, torch.Tensor), (
        "Expected `uv_mask` to be a tensor. " f"{type(uv_mask)=}"
    )
    assert isinstance(raster_face_indices, torch.Tensor), (
        "Expected `raster_face_indices` to be a tensor. "
        f"{type(raster_face_indices)=}"
    )

    uv_raster_triangle_indices = rast_out[..., 3].to(dtype=torch.long) - 1
    uv_visible = uv_raster_triangle_indices >= 0
    uv_weight = torch.zeros_like(uv_mask)
    if torch.any(uv_visible):
        uv_weight_values = face_weight[
            raster_face_indices[uv_raster_triangle_indices[uv_visible]]
        ]
        uv_weight[uv_visible.unsqueeze(-1)] = uv_weight_values.reshape(-1)
    uv_weight = uv_weight.clamp(min=0.0) * uv_mask
    return uv_weight
