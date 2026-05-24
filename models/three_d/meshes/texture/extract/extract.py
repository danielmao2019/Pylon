"""Texture extraction utilities for generic triangle meshes."""

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map
from data.structures.three_d.mesh.texture.validate_vertex_color import (
    validate_vertex_color,
)
from models.three_d.meshes.texture.extract.camera_geometry import (
    _project_verts_to_image,
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


def _validate_rgb_image(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), (
        "Expected the RGB image to be a `torch.Tensor`. " f"{type(obj)=}"
    )
    assert obj.ndim in (3, 4), (
        "Expected the RGB image to be rank 3 or 4. " f"{obj.shape=}"
    )
    if obj.ndim == 3:
        assert obj.shape[0] == 3 or obj.shape[2] == 3, (
            "Expected a rank-3 RGB image to be CHW or HWC with three channels. "
            f"{obj.shape=}"
        )
        image_height = int(obj.shape[1] if obj.shape[0] == 3 else obj.shape[0])
        image_width = int(obj.shape[2] if obj.shape[0] == 3 else obj.shape[1])
    else:
        assert obj.shape[0] == 1, (
            "Expected a rank-4 RGB image to have batch size 1. " f"{obj.shape=}"
        )
        assert obj.shape[1] == 3 or obj.shape[3] == 3, (
            "Expected a rank-4 RGB image to be NCHW or NHWC with three channels. "
            f"{obj.shape=}"
        )
        image_height = int(obj.shape[2] if obj.shape[1] == 3 else obj.shape[1])
        image_width = int(obj.shape[3] if obj.shape[1] == 3 else obj.shape[2])
    assert image_height > 0 and image_width > 0, (
        "Expected the RGB image to have positive spatial resolution. " f"{obj.shape=}"
    )

    if obj.dtype == torch.uint8:
        return
    assert obj.dtype == torch.float32, (
        "Expected the RGB image to be either uint8 `[0, 255]` or float32 "
        "`[0, 1]`. "
        f"{obj.dtype=}"
    )
    assert torch.isfinite(obj).all(), (
        "Expected float32 RGB values to contain only finite values. "
        f"{obj.shape=} {obj.dtype=}"
    )
    min_value = float(obj.min().item())
    max_value = float(obj.max().item())
    assert min_value >= 0.0, (
        "Expected float32 RGB values to be at least 0. " f"{min_value=}"
    )
    assert max_value <= 1.0, (
        "Expected float32 RGB values to be at most 1. " f"{max_value=}"
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
        default_color: Fallback color for texels/verts without any valid observation.
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
                assert mesh[0].verts.shape == view_mesh.verts.shape, (
                    "Expected all per-view meshes to share one vertex layout. "
                    f"{mesh[0].verts.shape=} {view_mesh.verts.shape=}"
                )
                reference_is_uv = isinstance(mesh[0].texture, MeshTextureUVTextureMap)
                view_is_uv = isinstance(view_mesh.texture, MeshTextureUVTextureMap)
                assert reference_is_uv == view_is_uv, (
                    "Expected all per-view meshes to agree on whether they carry a "
                    "UV-texture-map texture. "
                    f"{reference_is_uv=} {view_is_uv=}"
                )
                if reference_is_uv:
                    assert (
                        mesh[0].texture.verts_uvs.shape
                        == view_mesh.texture.verts_uvs.shape
                    ), (
                        "Expected all per-view meshes to share one UV layout. "
                        f"{mesh[0].texture.verts_uvs.shape=} "
                        f"{view_mesh.texture.verts_uvs.shape=}"
                    )
                    assert mesh[0].texture.convention == view_mesh.texture.convention, (
                        "Expected all per-view meshes to share one UV convention. "
                        f"{mesh[0].texture.convention=} "
                        f"{view_mesh.texture.convention=}"
                    )
                    assert torch.equal(
                        mesh[0].texture.verts_uvs.detach().cpu(),
                        view_mesh.texture.verts_uvs.detach().cpu(),
                    ), (
                        "Expected all per-view meshes to share identical UV "
                        "coordinates. "
                        f"{mesh[0].texture.verts_uvs.shape=} "
                        f"{view_mesh.texture.verts_uvs.shape=}"
                    )
                    assert (
                        mesh[0].texture.faces_uvs.shape
                        == view_mesh.texture.faces_uvs.shape
                    ), (
                        "Expected all per-view meshes to share one UV-face layout. "
                        f"{mesh[0].texture.faces_uvs.shape=} "
                        f"{view_mesh.texture.faces_uvs.shape=}"
                    )
                    assert torch.equal(
                        mesh[0].texture.faces_uvs.detach().cpu(),
                        view_mesh.texture.faces_uvs.detach().cpu(),
                    ), (
                        "Expected all per-view meshes to share identical UV-face "
                        "indices. "
                        f"{mesh[0].texture.faces_uvs.shape=} "
                        f"{view_mesh.texture.faces_uvs.shape=}"
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
    extract_uv_texture_map = isinstance(meshes[0].texture, MeshTextureUVTextureMap)

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

    device = meshes[0].verts.device
    meshes = [view_mesh.to(device=device) for view_mesh in meshes]
    images_nchw = images_nchw.to(device=device)
    cameras = cameras.to(device=device, convention="opencv")

    if not extract_uv_texture_map:
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
        default_color: Fallback color for verts without valid observations.

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
        default_color: Fallback color for verts without valid observations.

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
        _validate_rgb_image(obj=image)
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
    """Project one image to verts and sample per-vertex RGB colors.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        default_color: Fallback color for invalid projections.

    Returns:
        Vertex RGB colors with shape [V, 3].
    """

    xy, _depth, _verts_camera, projection_valid = _project_verts_to_image(
        verts=mesh.verts,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )

    vertex_count = mesh.verts.shape[0]
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
        assert isinstance(meshes[0].texture, MeshTextureUVTextureMap), (
            "Expected the reference mesh to carry a `MeshTextureUVTextureMap` "
            "texture for UV extraction. "
            f"{type(meshes[0].texture)=}"
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
    texel_face_map = build_texel_face_map(
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
                texel_face_map=texel_face_map,
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
            _validate_rgb_image(obj=observation["texture"])
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
        _validate_rgb_image(obj=provisional_uv_texture_map)
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
    _validate_rgb_image(obj=uv_texture_map)
    has_weight = uv_denominator > 0.0
    uv_texture_map = torch.where(
        has_weight.expand_as(uv_texture_map),
        uv_numerator / (uv_denominator + 1e-6),
        uv_texture_map,
    )
    _validate_rgb_image(obj=uv_texture_map)
    return {
        "texture": uv_texture_map.contiguous(),
        "valid_mask": has_weight.to(dtype=torch.float32).contiguous(),
    }


def _extract_uv_texture_map_from_single_image(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    weights_cfg: Dict[str, Any],
    texel_face_map: Dict[str, torch.Tensor],
    texel_visibility_method: str = "v1",
    polygon_rast_method: str = "v2",
) -> Dict[str, torch.Tensor]:
    """Extract one-view UV texture observation and UV weight map.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        weights_cfg: One-view weighting configuration dictionary.
        texel_face_map: Texel -> mesh-face correspondence dict from
            `build_texel_face_map` (`"texel_face_index"` [T, T] int64,
            `"texel_face_barycentric"` [T, T, 3] float32).
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
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected `mesh` to carry a `MeshTextureUVTextureMap` texture for "
            "UV extraction. "
            f"{type(mesh.texture)=}"
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
        assert isinstance(texel_face_map, dict), (
            "Expected `texel_face_map` to be a dictionary. "
            f"{type(texel_face_map)=}"
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
        _validate_rgb_image(obj=image)
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

    faces_uvs_long = mesh.texture.faces_uvs.to(
        device=mesh.texture.verts_uvs.device, dtype=torch.long
    )
    face_verts_uvs = mesh.texture.verts_uvs[faces_uvs_long].to(
        dtype=torch.float32
    ).contiguous()

    if texel_visibility_method == "v1":
        uv_visibility_mask = compute_f_visibility_mask(
            verts=mesh.verts,
            faces=mesh.faces,
            face_verts_uvs=face_verts_uvs,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            texel_face_map=texel_face_map,
            polygon_rast_method=polygon_rast_method,
        )
    else:
        assert texel_visibility_method == "v2", (
            "Expected `texel_visibility_method` to be `v1` or `v2`. "
            f"{texel_visibility_method=}"
        )
        uv_visibility_mask = compute_f_visibility_mask_v2(
            verts=mesh.verts,
            faces=mesh.faces,
            face_verts_uvs=face_verts_uvs,
            camera=camera,
            image_height=int(image.shape[1]),
            image_width=int(image.shape[2]),
            texel_face_map=texel_face_map,
        )
    if weights == "normals":
        face_normals_weight = _compute_f_normals_weights(
            mesh=mesh,
            camera=camera,
            weights_cfg=weights_cfg,
        )
        uv_normals_weight = _rasterize_face_weights_to_uv(
            face_weight=face_normals_weight,
            texel_face_map=texel_face_map,
        )
        uv_weight = uv_visibility_mask * uv_normals_weight
    else:
        uv_weight = uv_visibility_mask

    uv_texture = _project_f_colors(
        mesh=mesh,
        image=image,
        camera=camera,
        texel_face_map=texel_face_map,
    )
    return {
        "texture": torch.flip(uv_texture, dims=[1]).contiguous(),
        "weight": torch.flip(uv_weight.to(dtype=torch.float32), dims=[1]).contiguous(),
    }


def _project_f_colors(
    mesh: Mesh,
    image: torch.Tensor,
    camera: Cameras,
    texel_face_map: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Project one image into UV space by per-texel face-barycentric interpolation of projected mesh vertices.

    Args:
        mesh: Extraction mesh.
        image: One RGB image [3, H, W].
        camera: One camera instance.
        texel_face_map: Texel -> mesh-face correspondence dict from
            `build_texel_face_map` (`"texel_face_index"` [T, T] int64,
            `"texel_face_barycentric"` [T, T, 3] float32).

    Returns:
        One-view UV RGB image with shape [1, T, T, 3].
    """

    def _validate_inputs() -> None:
        assert isinstance(texel_face_map, dict), (
            "Expected `texel_face_map` to be a dictionary. "
            f"{type(texel_face_map)=}"
        )
        assert "texel_face_index" in texel_face_map, (
            "Expected `texel_face_map` to contain `texel_face_index`. "
            f"{texel_face_map.keys()=}"
        )
        assert "texel_face_barycentric" in texel_face_map, (
            "Expected `texel_face_map` to contain `texel_face_barycentric`. "
            f"{texel_face_map.keys()=}"
        )

    _validate_inputs()

    def _interpolate_uv_texel_image_coords(
        projected_vertex_xy: torch.Tensor,
        texel_face_map: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Interpolate image-space coordinates at every texel as a barycentric weighted sum of the projected mesh vertices of the texel's owning face.

        Args:
            projected_vertex_xy: Image-space mesh-vertex coordinates `[V, 2]`.
            texel_face_map: Texel -> mesh-face correspondence dict.

        Returns:
            Interpolated image-space UV texel coordinates `[1, T, T, 2]`
            (zeros at unoccupied texels).
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

        texel_face_index = texel_face_map["texel_face_index"]
        texel_face_barycentric = texel_face_map["texel_face_barycentric"]
        clamped_face_index = texel_face_index.clamp(min=0)
        per_corner_xy = projected_vertex_xy[mesh.faces[clamped_face_index]]
        interpolated_xy = (
            per_corner_xy * texel_face_barycentric.unsqueeze(-1)
        ).sum(dim=-2)
        occupied_mask = (texel_face_index >= 0).unsqueeze(-1)
        interpolated_xy = torch.where(
            occupied_mask, interpolated_xy, torch.zeros_like(interpolated_xy)
        )
        return interpolated_xy.unsqueeze(0).contiguous()

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

    xy, _depth, _verts_camera, _valid = _project_verts_to_image(
        verts=mesh.verts,
        camera=camera,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
    )
    interpolated_uv_xy = _interpolate_uv_texel_image_coords(
        projected_vertex_xy=xy,
        texel_face_map=texel_face_map,
    )
    uv_texture = _sample_uv_texel_colors_from_source_image(
        interpolated_uv_xy=interpolated_uv_xy,
        image=image,
    )
    _validate_rgb_image(obj=uv_texture)
    return uv_texture.contiguous()


def _rasterize_face_weights_to_uv(
    face_weight: torch.Tensor,
    texel_face_map: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Map per-face weights to per-UV-texel weights by gathering on `texel_face_index`, zero at unoccupied texels.

    Args:
        face_weight: Per-face weights [F].
        texel_face_map: Texel -> mesh-face correspondence dict from
            `build_texel_face_map` with key `"texel_face_index"` [T, T] int64
            (`-1` at unoccupied texels).

    Returns:
        UV weight map [1, T, T, 1] with non-negative values.
    """

    def _validate_inputs() -> None:
        assert isinstance(face_weight, torch.Tensor), (
            "Expected `face_weight` to be a tensor. " f"{type(face_weight)=}"
        )
        assert isinstance(texel_face_map, dict), (
            "Expected `texel_face_map` to be a dictionary. "
            f"{type(texel_face_map)=}"
        )
        assert "texel_face_index" in texel_face_map, (
            "Expected `texel_face_map` to contain `texel_face_index`. "
            f"{texel_face_map.keys()=}"
        )
        assert isinstance(texel_face_map["texel_face_index"], torch.Tensor), (
            "Expected `texel_face_index` to be a tensor. "
            f"{type(texel_face_map['texel_face_index'])=}"
        )
        assert texel_face_map["texel_face_index"].ndim == 2, (
            "Expected `texel_face_index` to have shape `[T, T]`. "
            f"{texel_face_map['texel_face_index'].shape=}"
        )
        assert face_weight.ndim == 1, (
            "Expected `face_weight` to have shape `[F]`. " f"{face_weight.shape=}"
        )

    _validate_inputs()

    texel_face_index = texel_face_map["texel_face_index"]
    occupied_mask = texel_face_index >= 0
    gathered = face_weight[texel_face_index.clamp(min=0)]
    uv_weight = torch.where(occupied_mask, gathered, torch.zeros_like(gathered))
    return uv_weight.clamp(min=0.0).unsqueeze(0).unsqueeze(-1).contiguous()

