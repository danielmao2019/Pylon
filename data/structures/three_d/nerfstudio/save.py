import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras


def save_intrinsic_params(params: Dict[str, float | int]) -> Dict[str, Any]:
    keys = ["fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2"]
    return {key: params[key] for key in keys}


def save_resolution(resolution: Tuple[int, int]) -> Dict[str, Any]:
    height, width = resolution
    return {"h": height, "w": width}


def save_camera_model(camera_model: str) -> Dict[str, Any]:
    return {"camera_model": camera_model}


def save_applied_transform(transform: np.ndarray) -> Dict[str, Any]:
    return {"applied_transform": transform.tolist()}


def save_ply_file_path(ply_file_path: str) -> Dict[str, Any]:
    return {"ply_file_path": ply_file_path}


def save_cameras(
    cameras: Cameras | List[Camera], modalities: Optional[List[str]] = None
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    include_masks = modalities is not None and "masks" in modalities
    for camera in cameras:
        assert camera.name is not None, "Camera name required to save transforms.json"
        frame_entry: Dict[str, Any] = {
            "file_path": f"images/{camera.name}.png",
            "transform_matrix": camera.extrinsics.detach().cpu().tolist(),
        }
        if camera.id is not None:
            frame_entry["colmap_image_id"] = camera.id
        if include_masks:
            frame_entry["mask_path"] = f"masks/{camera.name}.png"
        frames.append(frame_entry)
    frames.sort(key=lambda entry: entry["file_path"])
    return {"frames": frames}


def save_split_filenames(
    train: List[str] | None,
    val: List[str] | None,
    test: List[str] | None,
) -> Dict[str, Any]:
    if train is None:
        return {}
    return {
        "train_filenames": train,
        "val_filenames": val,
        "test_filenames": test,
    }


def save_nerfstudio_data(
    data: "NerfStudio_Data",
    filepath: str | Path,
) -> None:
    # Input validations
    assert data.__class__.__name__ == "NerfStudio_Data", f"{type(data)=}"
    assert (
        data.__class__.__module__
        == "data.structures.three_d.nerfstudio.nerfstudio_data"
    ), f"{type(data)=}"
    assert isinstance(filepath, (str, Path)), f"{type(filepath)=}"

    # Input normalizations
    path = Path(filepath)
    modalities = None
    if "modalities" in data.data:
        modalities = data.data["modalities"]
    intrinsic_params = data.intrinsic_params
    resolution = data.resolution
    camera_model = data.camera_model
    applied_transform = data.applied_transform
    ply_file_path = data.ply_file_path
    cameras = data.cameras
    train_filenames = data.train_filenames
    val_filenames = data.val_filenames
    test_filenames = data.test_filenames

    payload: Dict[str, Any] = {}
    payload.update(save_intrinsic_params(intrinsic_params))
    payload.update(save_resolution(resolution))
    payload.update(save_camera_model(camera_model))
    payload.update(save_applied_transform(applied_transform))
    payload.update(save_ply_file_path(ply_file_path))
    payload.update(save_cameras(cameras, modalities=modalities))
    payload.update(
        save_split_filenames(
            train=train_filenames,
            val=val_filenames,
            test=test_filenames,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
