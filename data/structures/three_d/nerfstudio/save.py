import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.nerfstudio.validate import MODALITY_SPECS


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
    cameras: Cameras | List[Camera],
    filenames: List[str],
    modalities: List[str],
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for camera, filename in zip(cameras, filenames, strict=True):
        assert camera.name is not None, "Camera name required to save transforms.json"
        assert camera.name == filename, f"{camera.name=} {filename=}"
        frame_entry: Dict[str, Any] = {
            "transform_matrix": camera.extrinsics.detach().cpu().tolist(),
        }
        if camera.id is not None:
            frame_entry["colmap_im_id"] = camera.id
        for modality in modalities:
            modality_key, modality_folder = MODALITY_SPECS[modality]
            frame_entry[modality_key] = f"{modality_folder}/{filename}.png"
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
    intrinsic_params = data.intrinsic_params
    resolution = data.resolution
    camera_model = data.camera_model
    applied_transform = data.applied_transform
    ply_file_path = data.ply_file_path
    cameras = data.cameras
    modalities = data.modalities
    filenames = data.filenames
    train_filenames = data.train_filenames
    val_filenames = data.val_filenames
    test_filenames = data.test_filenames

    payload: Dict[str, Any] = {}
    payload.update(save_intrinsic_params(intrinsic_params))
    payload.update(save_resolution(resolution))
    payload.update(save_camera_model(camera_model))
    payload.update(save_applied_transform(applied_transform))
    payload.update(save_ply_file_path(ply_file_path))
    payload.update(
        save_cameras(
            cameras=cameras,
            filenames=filenames,
            modalities=modalities,
        )
    )
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
