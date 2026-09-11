from typing import Any, Dict, Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.render_depth import render_depth_from_point_cloud
from models.three_d.point_cloud.render.render_normal import (
    render_normal_from_point_cloud_2d,
    render_normal_from_point_cloud_3d,
)
from models.three_d.point_cloud.render.render_rgb import render_rgb_from_point_cloud
from models.three_d.point_cloud.render.render_segmentation import (
    render_segmentation_from_point_cloud,
)

# Every public point-cloud entry main and this branch both carry.
RENDERERS: Tuple[str, ...] = ("depth", "rgb", "segmentation", "normal_3d", "normal_2d")
# Odd and even, so both kernel shapes are reached.
POINT_SIZES: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
RETURN_MASK_OPTIONS: Tuple[bool, ...] = (False, True)
# cpu, followed by cuda:0 when cuda is available; the indexed spelling, since main's Camera compares a bare cuda unequal to its components' cuda:0.
DEVICES: Tuple[torch.device, ...] = (torch.device("cpu"),) + (
    (torch.device("cuda:0"),) if torch.cuda.is_available() else ()
)


def build_point_cloud(scene: Dict[str, Any], device: torch.device) -> PointCloud:
    """Rebuilds a scene's cloud from its stored tensors through the constructor both checkouts share, so main and this branch render the same points.

    Args:
        scene: Scene dict holding the cloud as cpu torch.Tensors: "xyz" [N, 3] float32 world-space points, "rgb" [N, 3] colours (float32 in [0, 1] or uint8 in [0, 255]), "labels" [N] int64 segmentation labels and "normals" [N, 3] float32 world-space unit normals.
        device: Device the cloud is rebuilt on.

    Returns:
        The PointCloud whose xyz and rgb / labels / normals fields live on device.
    """
    return PointCloud(
        xyz=scene["xyz"].to(device),
        data={
            "rgb": scene["rgb"].to(device),
            "labels": scene["labels"].to(device),
            "normals": scene["normals"].to(device),
        },
    )


def build_camera(
    scene: Dict[str, Any], camera_index: int, device: torch.device
) -> Camera:
    """Rebuilds one of a scene's cameras from its stored tensors through the constructors both checkouts share.

    Args:
        scene: Scene dict holding the intrinsics "model" and "intr_convention", the pose "extr_convention", and "cameras", a list of {"params": 0-dim float32 cpu torch.Tensors fx / fy / cx / cy / h / w stated in scene["intr_convention"], "extrinsics": [4, 4] float32 cpu camera-to-world torch.Tensor in scene["extr_convention"]}.
        camera_index: Index into scene["cameras"] of the camera to rebuild.
        device: Device the camera is rebuilt on.

    Returns:
        The Camera whose intrinsics and extrinsics live on device, extrinsics kept in scene["extr_convention"].
    """
    camera_spec = scene["cameras"][camera_index]
    intrinsics = build_camera_intrinsics(
        model=scene["model"],
        params=camera_spec["params"],
        intr_convention=scene["intr_convention"],
        device=device,
    )
    extrinsics = CameraExtrinsics(
        extrinsics=camera_spec["extrinsics"],
        extr_convention=scene["extr_convention"],
        device=device,
    )
    return Camera(intrinsics=intrinsics, extrinsics=extrinsics, device=device)


def render_single_camera(
    renderer: str,
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    return_mask: bool,
    point_size: float,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Renders one camera through the named entry with the same keyword arguments on either checkout, each entry keeping its own default background.

    Args:
        renderer: One of RENDERERS, naming the public point-cloud entry to call.
        pc: PointCloud carrying xyz plus the "rgb", "labels" and "normals" fields the entries read.
        camera: The Camera to render through.
        resolution: Render resolution as an (H, W) tuple.
        return_mask: Whether the entry also returns its [H, W] bool valid-pixel mask.
        point_size: Diameter in pixels each rendered point is drawn with.

    Returns:
        The entry's output: the rendered map ([H, W] float32 depth, [3, H, W] float32 rgb, [H, W] int64 segmentation or [3, H, W] float32 normals), or the (map, [H, W] bool mask) tuple when return_mask is True.
    """

    def _validate_inputs() -> None:
        assert renderer in RENDERERS, (
            "Expected renderer to name one of the entries both checkouts carry. "
            f"{renderer=} {RENDERERS=}"
        )

    _validate_inputs()

    if renderer == "depth":
        return render_depth_from_point_cloud(
            pc=pc,
            camera=camera,
            resolution=resolution,
            return_mask=return_mask,
            point_size=point_size,
        )
    elif renderer == "rgb":
        return render_rgb_from_point_cloud(
            pc=pc,
            camera=camera,
            resolution=resolution,
            return_mask=return_mask,
            point_size=point_size,
        )
    elif renderer == "segmentation":
        return render_segmentation_from_point_cloud(
            pc=pc,
            key="labels",
            camera=camera,
            resolution=resolution,
            return_mask=return_mask,
            point_size=point_size,
        )
    elif renderer == "normal_3d":
        return render_normal_from_point_cloud_3d(
            pc=pc,
            camera=camera,
            resolution=resolution,
            return_mask=return_mask,
            point_size=point_size,
        )
    else:
        return render_normal_from_point_cloud_2d(
            pc=pc,
            camera=camera,
            resolution=resolution,
            return_mask=return_mask,
            point_size=point_size,
        )
